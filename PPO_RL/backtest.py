import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PPO_RL.config import (
    INITIAL_CAPITAL,
    TRANSACTION_COST,
    ACTION_THRESHOLD,
    RESULTS_DIR
)

from .environment import env_reset, env_step, get_observation
from PPO_RL.networks import sample_action


# ─────────────────────────────────────────────────────────────
# BACKTEST
# ─────────────────────────────────────────────────────────────

def backtest(pair_test_data, net):
    """Run PPO actor on held-out test pairs."""

    all_results = []

    for (test_data, pair_type, pair_info) in pair_test_data:
        features, prices_a, prices_b, dates = test_data
        stock_a, stock_b, _, gnn_strength = pair_info

        if len(features) < 10:
            continue

        state = env_reset(
            prices_a,
            prices_b,
            features,
            pair_type,
            INITIAL_CAPITAL,
            TRANSACTION_COST
        )

        obs = get_observation(state, 0)
        done = False

        while not done:
            action, _, _, _ = sample_action(net, obs, deterministic=True)

            obs, reward, done, info = env_step(
                state,
                action,
                ACTION_THRESHOLD
            )

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


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

def compute_metrics(equity_curve, trade_log):
    equity = np.array(equity_curve, dtype=np.float64)

    returns = np.diff(equity) / np.where(equity[:-1] != 0, equity[:-1], 1.0)
    returns = np.nan_to_num(returns)

    sharpe = (
        np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        if len(returns) > 1
        else 0.0
    )

    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / np.where(peak != 0, peak, 1.0)
    max_dd = np.max(drawdown) if len(drawdown) > 0 else 0.0

    total_return = (
        (equity[-1] - equity[0]) / equity[0]
        if equity[0] != 0
        else 0.0
    )

    closed = [t for t in trade_log if t.get("action") == "CLOSE"]
    wins = sum(1 for t in closed if t.get("realised_pnl", 0) > 0)
    win_rate = wins / len(closed) if closed else 0.0

    return {
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "total_return": total_return,
        "win_rate": win_rate,
        "num_trades": len(closed),
    }


# ─────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────

def save_results(all_results, history, save_dir=None):
    """Save PPO backtest results safely (no KeyErrors)."""

    save_dir = save_dir or RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    # ─────────────────────────────────────────────
    # 1. SUMMARY CSV
    # ─────────────────────────────────────────────
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

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(save_dir, "backtest_summary.csv"),
        index=False
    )

    print(f"Saved backtest summary → {save_dir}/backtest_summary.csv")

    # ─────────────────────────────────────────────
    # 2. TRADE LOG CSV
    # ─────────────────────────────────────────────
    trade_rows = []

    for r in all_results:
        pair_name = f"{r['stock_a']}-{r['stock_b']}"

        for t in r["trade_log"]:
            action = t.get("action")

            if action == "OPEN_LONG":
                bought, sold = r["stock_a"], r["stock_b"]
            elif action == "OPEN_SHORT":
                bought, sold = r["stock_b"], r["stock_a"]
            else:
                bought, sold = "CLOSE", "CLOSE"

            step = t.get("step", 0)
            timestamp = (
                r["dates"][step]
                if r.get("dates") is not None and step < len(r["dates"])
                else step
            )

            trade_rows.append({
                "Pair": pair_name,
                "Pair_Type": r["pair_type"],
                "Action": action,
                "Bought": bought,
                "Sold": sold,
                "Timestamp": timestamp,
                "Step": step,
                "Price_A": t.get("price_a"),
                "Price_B": t.get("price_b"),
                "Shares_A": t.get("shares_a"),
                "Shares_B": t.get("shares_b"),
                "Cost": t.get("cost"),
                "PnL": t.get("realised_pnl"),
                "Capital": t.get("capital_after"),
            })

    pd.DataFrame(trade_rows).to_csv(
        os.path.join(save_dir, "trade_log.csv"),
        index=False
    )

    print(f"Saved trade log → {save_dir}/trade_log.csv")

    # ─────────────────────────────────────────────
    # 3. TRAINING HISTORY
    # ─────────────────────────────────────────────
    episode_count = len(history.get("episode_rewards", []))

    hist_df = pd.DataFrame({
        "episode": list(range(1, episode_count + 1)),
        "reward": history.get("episode_rewards", []),
        "pnl": history.get("episode_pnl", []),
        "actor_loss": history.get("actor_loss", [0.0] * episode_count),
        "value_loss": history.get("value_loss", [0.0] * episode_count),
        "entropy": history.get("entropy", [0.0] * episode_count),
    })

    hist_df.to_csv(
        os.path.join(save_dir, "training_history.csv"),
        index=False
    )

    print(f"Saved training history → {save_dir}/training_history.csv")

# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────

def plot_performance(all_results, history, save_dir=None):
    """Safe PPO plotting (no KeyErrors, fully robust)."""

    save_dir = save_dir or RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    # ─────────────────────────────
    # TRAINING CURVES
    # ─────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PPO Training Progress")

    episodes = range(1, len(history.get("episode_rewards", [])) + 1)

    # Reward
    axes[0, 0].plot(episodes, history.get("episode_rewards", []))
    axes[0, 0].set_title("Reward")

    # PnL
    axes[0, 1].plot(episodes, history.get("episode_pnl", []))
    axes[0, 1].set_title("PnL")

    # Actor loss (SAFE)
    actor_loss = history.get("actor_loss", [0.0] * len(episodes))
    axes[1, 0].plot(episodes, actor_loss)
    axes[1, 0].set_title("Actor Loss")

    # Entropy
    entropy = history.get("entropy", [0.0] * len(episodes))
    axes[1, 1].plot(episodes, entropy)
    axes[1, 1].set_title("Entropy")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"))
    plt.close()

    print(f"Saved → training_curves.png")

    # ─────────────────────────────
    # EQUITY CURVES
    # ─────────────────────────────
    if not all_results:
        return

    top = sorted(all_results, key=lambda x: x["total_return"], reverse=True)[:5]

    fig, ax = plt.subplots(figsize=(12, 6))

    for r in top:
        ax.plot(r["equity_curve"], label=f"{r['stock_a']}-{r['stock_b']}")

    ax.axhline(y=INITIAL_CAPITAL, linestyle="--", color="gray")
    ax.set_title("Top Equity Curves")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "equity_curves.png"))
    plt.close()

    print(f"Saved → equity_curves.png")