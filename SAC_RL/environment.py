import numpy as np

def env_reset(prices_a, prices_b, features, pair_type, initial_capital, transaction_cost):
    """Initialise a new trading episode."""

    state = {
        "prices_a": prices_a,
        "prices_b": prices_b,
        "features": features,
        "pair_type": pair_type,
        "initial_capital": initial_capital,
        "capital": initial_capital,
        "transaction_cost": transaction_cost,
        "position": 0,          # -1 = short spread, 0 = flat, +1 = long spread
        "entry_price_a": 0.0,
        "entry_price_b": 0.0,
        "shares_a": 0.0,
        "shares_b": 0.0,
        "total_pnl": 0.0,
        "unrealised_pnl": 0.0,
        "step": 0,
        "max_steps": len(prices_a) - 1,
        "trade_log": [],
        "equity_curve": [initial_capital],
        "prev_position": 0,
    }
    return state


def _calculate_unrealised_pnl(state, step):
    """Compute unrealised PnL for the current open position."""

    if state["position"] == 0:
        return 0.0

    curr_a = state["prices_a"][step]
    curr_b = state["prices_b"][step]

    if state["pair_type"] == "positive":
        if state["position"] == 1:
            pnl_a = (curr_a - state["entry_price_a"]) * state["shares_a"]
            pnl_b = (state["entry_price_b"] - curr_b) * state["shares_b"]
        else:
            pnl_a = (state["entry_price_a"] - curr_a) * state["shares_a"]
            pnl_b = (curr_b - state["entry_price_b"]) * state["shares_b"]
    else:
        if state["position"] == 1:
            pnl_a = (curr_a - state["entry_price_a"]) * state["shares_a"]
            pnl_b = (curr_b - state["entry_price_b"]) * state["shares_b"]
        else:
            pnl_a = (state["entry_price_a"] - curr_a) * state["shares_a"]
            pnl_b = (state["entry_price_b"] - curr_b) * state["shares_b"]

    return pnl_a + pnl_b


def _open_position(state, step, direction):
    """Open a new position (direction = +1 or -1)."""

    price_a = state["prices_a"][step]
    price_b = state["prices_b"][step]

    half_cap = state["capital"] / 2.0
    shares_a = half_cap / price_a if price_a > 0 else 0.0
    shares_b = half_cap / price_b if price_b > 0 else 0.0

    cost = state["capital"] * state["transaction_cost"]
    state["capital"] -= cost

    state["position"] = direction
    state["entry_price_a"] = price_a
    state["entry_price_b"] = price_b
    state["shares_a"] = shares_a
    state["shares_b"] = shares_b

    state["trade_log"].append({
        "action": "OPEN_LONG" if direction == 1 else "OPEN_SHORT",
        "step": step,
        "price_a": price_a,
        "price_b": price_b,
        "shares_a": shares_a,
        "shares_b": shares_b,
        "capital_after": state["capital"],
        "cost": cost,
    })


def _close_position(state, step):
    """Close the current open position and realise PnL."""

    if state["position"] == 0:
        return 0.0

    realised = _calculate_unrealised_pnl(state, step)

    close_value = (state["shares_a"] * state["prices_a"][step] +
                   state["shares_b"] * state["prices_b"][step])
    cost = close_value * state["transaction_cost"]

    realised -= cost
    state["capital"] += realised
    state["total_pnl"] += realised

    state["trade_log"].append({
        "action": "CLOSE",
        "step": step,
        "price_a": state["prices_a"][step],
        "price_b": state["prices_b"][step],
        "realised_pnl": realised,
        "capital_after": state["capital"],
        "cost": cost,
    })

    state["position"] = 0
    state["entry_price_a"] = 0.0
    state["entry_price_b"] = 0.0
    state["shares_a"] = 0.0
    state["shares_b"] = 0.0

    return realised


def env_step(state, action, action_threshold):
    """ Execute one step in the trading environment."""

    step = state["step"]
    prev_equity = state["capital"] + _calculate_unrealised_pnl(state, step)
    state["prev_position"] = state["position"]

    if action > action_threshold:
        desired_pos = 1
    elif action < -action_threshold:
        desired_pos = -1
    else:
        desired_pos = 0

    # Execute position change
    position_changed = False
    if desired_pos != state["position"]:
        if state["position"] != 0:
            _close_position(state, step)
        if desired_pos != 0:
            _open_position(state, step, desired_pos)
        position_changed = True

    # Advance time
    state["step"] += 1
    next_step = state["step"]
    done = next_step >= state["max_steps"]

    # Force close at episode end
    if done and state["position"] != 0:
        _close_position(state, min(next_step, len(state["prices_a"]) - 1))

    # Compute reward
    safe_step = min(next_step, len(state["prices_a"]) - 1)
    current_equity = state["capital"] + _calculate_unrealised_pnl(state, safe_step)
    state["unrealised_pnl"] = _calculate_unrealised_pnl(state, safe_step)
    state["equity_curve"].append(current_equity)
    reward = (current_equity - prev_equity) / state["initial_capital"] * 100.0

    # Small penalty for position switching (discourages churning)
    if position_changed:
        reward -= 0.01

    obs = get_observation(state, min(next_step, len(state["features"]) - 1))
    info = {
        "equity": current_equity,
        "position": state["position"],
        "total_pnl": state["total_pnl"],
        "step": next_step,
    }
    return obs, reward, done, info


def get_observation(state, step):
    """Build the observation vector for the agent."""
    
    market_features = state["features"][step]

    pos = state["position"]
    pos_long = 1.0 if pos == 1 else 0.0
    pos_short = 1.0 if pos == -1 else 0.0
    pos_flat = 1.0 if pos == 0 else 0.0

    norm_pnl = state["unrealised_pnl"] / state["initial_capital"]
    norm_cap = state["capital"] / state["initial_capital"] - 1.0

    extra = np.array([pos_long, pos_short, pos_flat, norm_pnl, norm_cap], dtype=np.float64)

    obs = np.concatenate([market_features, extra])
    return obs.astype(np.float32)


def get_obs_dim():
    """Return the observation dimension: 11 market features + 5 state features."""
    return 16