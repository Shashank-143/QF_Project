import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from data_loader import load_data
from model_loader import load_actor
from inference_engine import run_inference


# =========================
# CONFIG
# =========================
MODEL_PATH = "models/actor.pth"
OBS_DIM = 16

CONFIG = {
    "initial_capital": 10000,
    "transaction_cost": 0.001,
    "action_threshold": 0.5
}

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRONTEND_DIR = os.path.join(REPO_ROOT, "frontend")
POSITIVE_PAIRS_PATH = os.path.join(REPO_ROOT, "notebooks", "discovered_positive_pairs.csv")
NEGATIVE_PAIRS_PATH = os.path.join(REPO_ROOT, "notebooks", "discovered_negative_pairs.csv")
BACKTEST_SUMMARY_PATH = os.path.join(REPO_ROOT, "SAC_RL", "results", "backtest_summary.csv")
PRICES_PATH = os.path.join(os.path.dirname(__file__), "data", "prices.csv")


# =========================
# INIT APP
# =========================
app = FastAPI(title="RL Trading Backend")


# Allow frontend (React etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# LOAD ON STARTUP
# =========================
print("Loading model and data...")

actor = load_actor(MODEL_PATH, OBS_DIM)
pair_data = load_data(
    prices_path=PRICES_PATH,
    pos_pairs_path=POSITIVE_PAIRS_PATH,
    neg_pairs_path=NEGATIVE_PAIRS_PATH,
    top_k=None,
)

print(f"Loaded {len(pair_data)} pairs")


def _calculate_overall_stats(results, initial_capital):
    trade_pnls = []
    for result in results:
        for trade in result.get("trades", []):
            if trade.get("action") == "CLOSE":
                pnl = trade.get("realised_pnl")
                if pnl is not None:
                    trade_pnls.append(float(pnl))

    total_pairs = len(results)
    total_initial = initial_capital * total_pairs
    total_pnl = float(sum(result.get("total_pnl", 0.0) for result in results))
    trade_pnl_total = float(sum(trade_pnls))
    profit_diff = total_pnl - trade_pnl_total
    profit_verified = abs(profit_diff) < 1e-6

    equity = total_initial
    peak = total_initial
    max_drawdown = 0.0
    for pnl in trade_pnls:
        equity += pnl
        if equity > peak:
            peak = equity
        drawdown = (equity - peak) / peak if peak else 0.0
        if drawdown < max_drawdown:
            max_drawdown = drawdown

    returns = [pnl / initial_capital for pnl in trade_pnls] if initial_capital else []
    sharpe_ratio = None
    alpha = None
    if len(returns) >= 2:
        mean = float(np.mean(returns))
        std = float(np.std(returns, ddof=1))
        alpha = mean
        if std > 0:
            sharpe_ratio = float(mean / std * np.sqrt(len(returns)))
    elif len(returns) == 1:
        alpha = float(returns[0])

    final_equity = total_initial + total_pnl
    return_pct = (total_pnl / total_initial * 100.0) if total_initial else 0.0

    return {
        "initial_capital": total_initial,
        "total_pnl": total_pnl,
        "final_equity": final_equity,
        "return_pct": return_pct,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "alpha": alpha,
        "trade_count": len(trade_pnls),
        "profit_verified": profit_verified,
        "profit_diff": profit_diff,
    }


def _load_pairs_csv(path, pair_type):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"CSV not found: {path}")

    df = pd.read_csv(path)
    required = {"Stock_A", "Stock_B", "Strength"}
    if not required.issubset(df.columns):
        raise HTTPException(status_code=500, detail="Pair CSV missing required columns")

    pairs = []
    for _, row in df.iterrows():
        stock_a = str(row["Stock_A"]).strip()
        stock_b = str(row["Stock_B"]).strip()
        pairs.append({
            "pair": f"{stock_a}-{stock_b}",
            "pair_type": pair_type,
            "gnn_strength": float(row["Strength"])
        })

    return pairs


def _load_backtest_summary(path):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"CSV not found: {path}")

    df = pd.read_csv(path)
    required = {
        "Stock_A", "Stock_B", "Pair_Type", "Total_PnL",
        "Sharpe_Ratio", "Max_Drawdown", "Num_Trades"
    }
    if not required.issubset(df.columns):
        raise HTTPException(status_code=500, detail="Backtest summary CSV missing required columns")

    summary = []
    for _, row in df.iterrows():
        stock_a = str(row["Stock_A"]).strip()
        stock_b = str(row["Stock_B"]).strip()
        summary.append({
            "pair": f"{stock_a}-{stock_b}",
            "pair_type": row["Pair_Type"],
            "pnl": float(row["Total_PnL"]),
            "trade_count": int(row["Num_Trades"]),
            "sharpe": float(row["Sharpe_Ratio"]),
            "max_drawdown": float(row["Max_Drawdown"]),
        })

    return summary


def _find_pair_data(pair_name):
    for test_data, pair_type, pair_info in pair_data:
        stock_a, stock_b, _, _ = pair_info
        if f"{stock_a}-{stock_b}" == pair_name:
            return (test_data, pair_type, pair_info)
    return None


# =========================
# ROUTES
# =========================

# -------------------------
# Run full inference
# -------------------------
@app.get("/run")
def run_model():
    results = run_inference(actor, pair_data, CONFIG)

    return {
        "num_pairs": len(results),
        "results": results
    }


# -------------------------
# Summary (lightweight)
# -------------------------
@app.get("/summary")
def summary():
    results = run_inference(actor, pair_data, CONFIG)
    stats = _calculate_overall_stats(results, CONFIG["initial_capital"])

    summary = []
    for r in results:
        summary.append({
            "pair": r["pair"],
            "pair_type": r["pair_type"],
            "pnl": r["total_pnl"],
            "equity": r["final_equity"],
            "trade_count": r.get("trade_count", 0),
            "gnn_strength": r.get("gnn_strength")
        })

    return {"summary": summary, "stats": stats}


@app.get("/pairs/positive")
def positive_pairs():
    return {"pairs": _load_pairs_csv(POSITIVE_PAIRS_PATH, "positive")}


@app.get("/pairs/negative")
def negative_pairs():
    return {"pairs": _load_pairs_csv(NEGATIVE_PAIRS_PATH, "negative")}


@app.get("/backtest_summary")
def backtest_summary():
    return {"summary": _load_backtest_summary(BACKTEST_SUMMARY_PATH)}


# -------------------------
# Single pair detail
# -------------------------
@app.get("/pair/{pair_name}")
def get_pair(pair_name: str):
    pair_item = _find_pair_data(pair_name)
    if not pair_item:
        return {"error": "Pair not found"}

    result = run_inference(actor, [pair_item], CONFIG)
    return result[0] if result else {"error": "Pair not found"}


# -------------------------
# Reload data (optional)
# -------------------------
@app.get("/reload")
def reload_data():
    global pair_data
    pair_data = load_data(
        prices_path=PRICES_PATH,
        pos_pairs_path=POSITIVE_PAIRS_PATH,
        neg_pairs_path=NEGATIVE_PAIRS_PATH,
        top_k=None,
    )

    return {
        "message": "Data reloaded",
        "pairs_loaded": len(pair_data)
    }


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
