import numpy as np
from environment import env_reset, env_step, get_observation
from networks import sample_action


def _strip_trade_times(trade_log):
    return [{k: v for k, v in trade.items() if k != "step"} for trade in trade_log]


def _count_trades(trade_log):
    return sum(1 for trade in trade_log if trade.get("action") == "CLOSE")


def _last_trade_details(trade_log):
    last_open = None
    last_close = None

    for trade in trade_log:
        action = trade.get("action")
        if action in ("OPEN_LONG", "OPEN_SHORT"):
            last_open = trade
        elif action == "CLOSE":
            last_close = trade

    details = {
        "shares_a": None,
        "shares_b": None,
        "entry_price_a": None,
        "entry_price_b": None,
        "exit_price_a": None,
        "exit_price_b": None,
        "total_cost": None,
        "capital_after": None,
        "action": "HOLD",
    }

    open_cost = None
    close_cost = None

    if last_open:
        details["shares_a"] = float(last_open.get("shares_a", 0.0))
        details["shares_b"] = float(last_open.get("shares_b", 0.0))
        details["entry_price_a"] = float(last_open.get("price_a", 0.0))
        details["entry_price_b"] = float(last_open.get("price_b", 0.0))
        details["action"] = "LONG" if last_open.get("action") == "OPEN_LONG" else "SHORT"
        open_cost = last_open.get("cost")

    if last_close:
        details["exit_price_a"] = float(last_close.get("price_a", 0.0))
        details["exit_price_b"] = float(last_close.get("price_b", 0.0))
        details["capital_after"] = float(last_close.get("capital_after", 0.0))
        close_cost = last_close.get("cost")

    if open_cost is not None or close_cost is not None:
        details["total_cost"] = float((open_cost or 0.0) + (close_cost or 0.0))

    return details


def run_inference(actor, pair_data, config):
    results = []

    for (test_data, pair_type, pair_info) in pair_data:
        features, prices_a, prices_b, dates = test_data
        stock_a, stock_b, _, strength = pair_info

        state = env_reset(
            prices_a,
            prices_b,
            features,
            pair_type,
            config["initial_capital"],
            config["transaction_cost"]
        )

        obs = get_observation(state, 0 )
        done = False

        while not done:
            action = sample_action(actor, obs, deterministic=True)
            obs, reward, done, info = env_step(state, action, config["action_threshold"])

        trade_log = state["trade_log"]
        trade_details = _last_trade_details(trade_log)
        results.append({
            "pair": f"{stock_a}-{stock_b}",
            "pair_type": pair_type,
            "gnn_strength": float(strength),
            "final_equity": float(state["equity_curve"][-1]),
            "total_pnl": float(state["total_pnl"]),
            "trade_count": _count_trades(trade_log),
            "trades": _strip_trade_times(trade_log),
            "shares_a": trade_details["shares_a"],
            "shares_b": trade_details["shares_b"],
            "entry_price_a": trade_details["entry_price_a"],
            "entry_price_b": trade_details["entry_price_b"],
            "exit_price_a": trade_details["exit_price_a"],
            "exit_price_b": trade_details["exit_price_b"],
            "total_cost": trade_details["total_cost"],
            "capital_after": trade_details["capital_after"],
            "action": trade_details["action"],
        })

    return results
