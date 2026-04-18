import pandas as pd
import numpy as np


# =========================
# LOAD PAIRS (GNN OUTPUT)
# =========================
def load_pairs(pos_path, neg_path, strength_threshold=1e-6, top_k=None):
    """
    Load positive + negative pairs from CSVs.

    Returns:
        List of tuples:
        (stock_a, stock_b, pair_type, strength)
    """

    pos_df = pd.read_csv(pos_path)
    neg_df = pd.read_csv(neg_path)

    pairs = []

    # Positive pairs
    for _, row in pos_df.iterrows():
        strength = float(row["Strength"])
        if abs(strength) < strength_threshold:
            continue
        pairs.append((row["Stock_A"], row["Stock_B"], "positive", strength))

    # Negative pairs
    for _, row in neg_df.iterrows():
        strength = float(row["Strength"])
        if abs(strength) < strength_threshold:
            continue
        pairs.append((row["Stock_A"], row["Stock_B"], "negative", strength))

    # Sort by strength (descending)
    pairs = sorted(pairs, key=lambda x: x[3], reverse=True)

    if top_k:
        pairs = pairs[:top_k]

    return pairs


# =========================
# LOAD PRICE DATA
# =========================
def load_price_data(csv_path):
    """
    Load merged stock price data.
    """

    df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    df = df.sort_index()
    df = df.ffill().bfill()

    return df


# =========================
# FEATURE ENGINEERING
# =========================
def compute_pair_features(prices, stock_a, stock_b, gnn_strength, window=20):
    """
    Compute features for a stock pair.
    Returns:
        features, prices_a, prices_b, dates
    """

    pa = prices[stock_a].values.astype(np.float64)
    pb = prices[stock_b].values.astype(np.float64)

    # Returns
    ret_a = np.diff(pa) / (pa[:-1] + 1e-8)
    ret_b = np.diff(pb) / (pb[:-1] + 1e-8)

    # Align lengths
    pa = pa[1:]
    pb = pb[1:]
    dates = prices.index[1:]

    # Spread (normalized)
    mean_price = (pa + pb) / 2.0
    spread = (pa - pb) / (mean_price + 1e-8)

    spread_series = pd.Series(spread)

    # Rolling stats
    roll_mean = spread_series.rolling(window, min_periods=1).mean().values
    roll_std = spread_series.rolling(window, min_periods=1).std().values
    roll_std = np.where(roll_std == 0, 1e-8, roll_std)

    # Z-score
    spread_z = (spread - roll_mean) / roll_std

    # Correlation
    ret_a_series = pd.Series(ret_a)
    ret_b_series = pd.Series(ret_b)
    rolling_corr = ret_a_series.rolling(window, min_periods=1).corr(ret_b_series).values
    rolling_corr = np.nan_to_num(rolling_corr)

    # Volatility
    vol_a = ret_a_series.rolling(window, min_periods=1).std().values
    vol_b = ret_b_series.rolling(window, min_periods=1).std().values

    # Momentum
    spread_momentum = spread_series.diff(window).values
    spread_momentum = np.nan_to_num(spread_momentum)

    # Mean reversion signal
    mean_reversion = spread - roll_mean

    # GNN strength (constant feature)
    gnn_feat = np.full_like(spread, gnn_strength)

    # Final feature matrix (11 features)
    features = np.column_stack([
        spread,
        spread_z,
        rolling_corr,
        gnn_feat,
        ret_a,
        ret_b,
        vol_a,
        vol_b,
        spread_momentum,
        mean_reversion,
        roll_mean,
    ])

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features, pa, pb, dates


# =========================
# PREPARE ALL PAIRS
# =========================
def prepare_pair_data(df, pairs):
    """
    Convert raw pairs + price data into RL-ready format.

    Returns:
        List of:
        (test_data, pair_type, pair_info)
    """

    pair_data = []

    for stock_a, stock_b, pair_type, strength in pairs:
        if stock_a not in df.columns or stock_b not in df.columns:
            continue

        try:
            features, pa, pb, dates = compute_pair_features(
                df, stock_a, stock_b, strength
            )

            # Skip very small datasets
            if len(features) < 30:
                continue

            test_data = (features, pa, pb, dates)
            pair_info = (stock_a, stock_b, None, strength)

            pair_data.append((test_data, pair_type, pair_info))

        except Exception as e:
            print(f"Skipping pair {stock_a}-{stock_b}: {e}")

    return pair_data


# =========================
# MAIN LOADER FUNCTION
# =========================
def load_data(
    prices_path="data/prices.csv",
    pos_pairs_path="data/discovered_positive_pairs.csv",
    neg_pairs_path="data/discovered_negative_pairs.csv",
    top_k=10
):
    """
    Main function used by backend.

    Returns:
        pair_data ready for inference
    """

    df = load_price_data(prices_path)

    pairs = load_pairs(
        pos_pairs_path,
        neg_pairs_path,
        top_k=top_k
    )

    pair_data = prepare_pair_data(df, pairs)

    print(f"Loaded {len(pair_data)} valid pairs for inference")

    return pair_data
