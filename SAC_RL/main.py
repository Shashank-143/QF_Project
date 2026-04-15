import sys
import os
import torch
#sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import POSITIVE_PAIRS_PATH, NEGATIVE_PAIRS_PATH, PRICE_DATA_PATH, RESULTS_DIR
from data_loader import load_pairs, load_price_data
from train import train_sac
from backtest import backtest, save_results, plot_performance


def main():
    print("SAC Pairs Trading Agent")

    print("\n[1/4] Loading pairs and price data...")
    pairs = load_pairs(POSITIVE_PAIRS_PATH, NEGATIVE_PAIRS_PATH)
    print(f"Loaded {len(pairs)} pairs ({len([p for p in pairs if p[2]=='positive'])} positive, "
          f"{len([p for p in pairs if p[2]=='negative'])} negative)")

    price_data = load_price_data(PRICE_DATA_PATH)
    print(f"Price data: {price_data.shape[0]} days, {price_data.shape[1]} stocks")
    print(f"Date range: {price_data.index[0].date()} to {price_data.index[-1].date()}")

    print("\n[2/4] Training SAC agent...")
    actor, history, pair_test_data = train_sac(pairs, price_data)

    print("\n[3/4] Backtesting on held-out test data...")
    results = backtest(pair_test_data, actor)
    print(f"  Backtested {len(results)} pairs")

    if results:
        total_pnl = sum(r["total_pnl"] for r in results)
        avg_return = sum(r["total_return"] for r in results) / len(results) * 100
        avg_sharpe = sum(r["sharpe_ratio"] for r in results) / len(results)
        print(f"\nPortfolio Total PnL:  Rs.{total_pnl:.2f}")
        print(f"Avg Return per Pair: {avg_return:.2f}%")
        print(f"Avg Sharpe Ratio:    {avg_sharpe:.2f}")

    print("\n[4/4] Saving results and plots...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_results(results, history, RESULTS_DIR)
    plot_performance(results, history, RESULTS_DIR)

    model_path = os.path.join(RESULTS_DIR, "trained_actor.pth")
    torch.save({
        "net": actor["net"].state_dict(),
        "mean": actor["mean"].state_dict(),
        "log_std": actor["log_std"].state_dict(),
    }, model_path)
    print(f"  Saved trained actor to {model_path}")
    print("  Done! Check results in:", RESULTS_DIR)

if __name__ == "__main__":
    main()