import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import INITIAL_CAPITAL, TRANSACTION_COST, ACTION_THRESHOLD, RESULTS_DIR
from environment import env_reset, env_step, get_observation
from networks import sample_action


def backtest(pair_test_data, actor):
    """Run the trained actor on held-out test data for every pair."""

    all_results = []

    for (test_data, pair_type, pair_info) in pair_test_data:
        features, prices_a, prices_b, dates = test_data
        stock_a, stock_b, _, gnn_strength = pair_info

        if len(features) < 10:
            continue

        state = env_reset(prices_a, prices_b, features, pair_type, INITIAL_CAPITAL, TRANSACTION_COST)
        obs = get_observation(state, 0)
        done = False

        while not done:
            action = sample_action(actor, obs, deterministic=True)
            obs, reward, done, info = env_step(state, action, ACTION_THRESHOLD)

        metrics = compute_metrics(state["equity_curve"], state["trade_log"])

        all_results.append({
            "stock_a": stock_a,
            "stock_b": stock_b,
            "pair_type": pair_type,
            "gnn_strength": gnn_strength,
            "total_pnl": state["total_pnl"],
            "final_equity": state["equity_curve"][-1],
            "equity_curve": state["equity_curve"],
            "trade_log": state["trade_log"],
            "dates": dates,
            **metrics,
        })

    return all_results


def compute_metrics(equity_curve, trade_log):
    """Compute performance metrics from an equity curve."""
    
    equity = np.array(equity_curve, dtype=np.float64)
    returns = np.diff(equity) / np.where(equity[:-1] != 0, equity[:-1], 1.0)
    returns = np.nan_to_num(returns, nan=0.0)

    # Sharpe ratio (annualised, assuming ~252 trading days)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / np.where(peak != 0, peak, 1.0)
    max_dd = np.max(drawdown) if len(drawdown) > 0 else 0.0
    total_ret = (equity[-1] - equity[0]) / equity[0] if equity[0] != 0 else 0.0

    closed = [t for t in trade_log if t.get("action") == "CLOSE"]
    wins = sum(1 for t in closed if t.get("realised_pnl", 0) > 0)
    win_rate = wins / len(closed) if closed else 0.0

    return {
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "total_return": total_ret,
        "win_rate": win_rate,
        "num_trades": len(closed),
    }


def save_results(all_results, history, save_dir=None):
    """Save trade logs, metrics summary, and training history as CSVs."""
    save_dir = save_dir or RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "Stock_A": r["stock_a"],
            "Stock_B": r["stock_b"],
            "Pair_Type": r["pair_type"],
            "GNN_Strength": r["gnn_strength"],
            "Total_PnL": r["total_pnl"],
            "Total_Return": r["total_return"],
            "Sharpe_Ratio": r["sharpe_ratio"],
            "Max_Drawdown": r["max_drawdown"],
            "Win_Rate": r["win_rate"],
            "Num_Trades": r["num_trades"],
            "Final_Equity": r["final_equity"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(save_dir, "backtest_summary.csv"), index=False)
    print(f"Saved backtest summary to {save_dir}/backtest_summary.csv")

    combined_rows = []
    for r in all_results:
        stock_a, stock_b = r["stock_a"], r["stock_b"]
        pair_type = r["pair_type"]
        pair_name = f"{stock_a}-{stock_b}"

        for t in r["trade_log"]:
            action = t["action"]

            if action == "OPEN_LONG":
                if pair_type == "positive":
                    bought, sold = stock_a, stock_b
                else:
                    bought, sold = f"{stock_a}+{stock_b}", "—"

            elif action == "OPEN_SHORT":
                if pair_type == "positive":
                    bought, sold = stock_b, stock_a
                else:
                    bought, sold = "—", f"{stock_a}+{stock_b}"
            else:
                bought, sold = "CLOSE", "CLOSE"

            step = t["step"]
            timestamp = (r["dates"][step]
                         if "dates" in r and r["dates"] is not None
                            and step < len(r["dates"])
                         else step)

            combined_rows.append({
                "Pair": pair_name,
                "Pair_Type": pair_type,
                "Action": action,
                "Bought": bought,
                "Sold": sold,
                "Timestamp": timestamp,
                "Step": step,
                "Price_A": t.get("price_a"),
                "Price_B": t.get("price_b"),
                "Shares_A": t.get("shares_a"),
                "Shares_B": t.get("shares_b"),
                "Transaction_Cost": t.get("cost"),
                "Realised_PnL": t.get("realised_pnl"),
                "Capital_After": t.get("capital_after"),
            })

    if combined_rows:
        trades_df = pd.DataFrame(combined_rows)
        trades_df.to_csv(os.path.join(save_dir, "trade_log.csv"), index=False)
        print(f"Saved combined trade log ({len(combined_rows)} entries) "
              f"to {save_dir}/trade_log.csv")

    hist_df = pd.DataFrame({
        "episode": list(range(1, len(history["episode_rewards"]) + 1)),
        "reward": history["episode_rewards"],
        "pnl": history["episode_pnl"],
        "critic_loss": history["critic_loss"],
        "actor_loss": history["actor_loss"],
        "alpha": history["alpha"],
    })
    hist_df.to_csv(os.path.join(save_dir, "training_history.csv"), index=False)
    print(f"Saved training history to {save_dir}/training_history.csv")


def plot_performance(all_results, history, save_dir=None):
    """Generate and save performance plots."""
    
    save_dir = save_dir or RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    # Training curves 
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SAC Training Progress", fontsize=14)
    episodes = range(1, len(history["episode_rewards"]) + 1)
    axes[0, 0].plot(episodes, history["episode_rewards"], alpha=0.3, color="blue")
    if len(history["episode_rewards"]) >= 10:
        smoothed = pd.Series(history["episode_rewards"]).rolling(10).mean()
        axes[0, 0].plot(episodes, smoothed, color="blue", linewidth=2)
    axes[0, 0].set_title("Episode Reward")
    axes[0, 0].set_xlabel("Episode")

    axes[0, 1].plot(episodes, history["episode_pnl"], alpha=0.3, color="green")
    if len(history["episode_pnl"]) >= 10:
        smoothed = pd.Series(history["episode_pnl"]).rolling(10).mean()
        axes[0, 1].plot(episodes, smoothed, color="green", linewidth=2)
    axes[0, 1].set_title("Episode PnL (Rs.)")
    axes[0, 1].set_xlabel("Episode")

    axes[1, 0].plot(episodes, history["critic_loss"], color="red", alpha=0.5)
    axes[1, 0].set_title("Critic Loss")
    axes[1, 0].set_xlabel("Episode")

    axes[1, 1].plot(episodes, history["alpha"], color="purple")
    axes[1, 1].set_title("Temperature (α)")
    axes[1, 1].set_xlabel("Episode")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"Saved training curves to {save_dir}/training_curves.png")

    # Backtest equity curves (top 5 by return) 
    if not all_results:
        return

    sorted_results = sorted(all_results, key=lambda r: r["total_return"], reverse=True)
    top_n = min(5, len(sorted_results))
    fig, ax = plt.subplots(figsize=(12, 6))
    for r in sorted_results[:top_n]:
        label = f"{r['stock_a']}-{r['stock_b']} ({r['pair_type'][:3]})"
        ax.plot(r["equity_curve"], label=label, alpha=0.8)
    ax.axhline(y=INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5, label="Initial Capital")
    ax.set_title("Backtest Equity Curves (Top 5 Pairs)")
    ax.set_xlabel("Trading Steps")
    ax.set_ylabel("Equity (Rs.)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "equity_curves.png"), dpi=150)
    plt.close()
    print(f"Saved equity curves to {save_dir}/equity_curves.png")

    # Summary bar chart 
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    labels = [f"{r['stock_a'][:4]}-{r['stock_b'][:4]}" for r in sorted_results[:10]]
    returns = [r["total_return"] * 100 for r in sorted_results[:10]]
    sharpes = [r["sharpe_ratio"] for r in sorted_results[:10]]
    drawdowns = [r["max_drawdown"] * 100 for r in sorted_results[:10]]

    colors = ["green" if v > 0 else "red" for v in returns]
    axes[0].barh(labels, returns, color=colors)
    axes[0].set_title("Total Return (%)")

    axes[1].barh(labels, sharpes, color="steelblue")
    axes[1].set_title("Sharpe Ratio")

    axes[2].barh(labels, drawdowns, color="coral")
    axes[2].set_title("Max Drawdown (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_summary.png"), dpi=150)
    plt.close()
    print(f"Saved metrics summary to {save_dir}/metrics_summary.png")