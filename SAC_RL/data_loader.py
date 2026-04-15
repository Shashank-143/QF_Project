import numpy as np
import pandas as pd


def load_pairs(pos_path, neg_path):
    """ Load GNN-discovered pairs from CSVs."""

    pos_df = pd.read_csv(pos_path)
    neg_df = pd.read_csv(neg_path)

    pairs = []
    for _, row in pos_df.iterrows():
        pairs.append((row["Stock_A"], row["Stock_B"], "positive", float(row["Strength"])))
    for _, row in neg_df.iterrows():
        pairs.append((row["Stock_A"], row["Stock_B"], "negative", float(row["Strength"])))

    return pairs


def load_price_data(csv_path):
    """ Load merged price data CSV. """

    df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    df = df.sort_index()
    df = df.ffill().bfill()
    return df


def compute_pair_features(prices, stock_a, stock_b, gnn_strength, window=20):
    """Compute observation features for a single pair."""

    pa = prices[stock_a].values.astype(np.float64)
    pb = prices[stock_b].values.astype(np.float64)
    ret_a = np.diff(pa) / pa[:-1]
    ret_b = np.diff(pb) / pb[:-1]

    # Align lengths (drop first element of prices to match returns)
    pa = pa[1:]
    pb = pb[1:]
    dates = prices.index[1:]

    # Price spread (normalised by mean price)
    mean_price = (pa + pb) / 2.0
    spread = (pa - pb) / np.where(mean_price != 0, mean_price, 1.0)

    # Rolling spread stats
    spread_series = pd.Series(spread)
    roll_mean = spread_series.rolling(window, min_periods=1).mean().values
    roll_std = spread_series.rolling(window, min_periods=1).std().values
    roll_std = np.where(roll_std == 0, 1e-8, roll_std)

    # Z-score of spread
    spread_zscore = (spread - roll_mean) / roll_std

    # Rolling correlation of returns
    ret_a_series = pd.Series(ret_a)
    ret_b_series = pd.Series(ret_b)
    rolling_corr = ret_a_series.rolling(window, min_periods=1).corr(ret_b_series).values
    rolling_corr = np.nan_to_num(rolling_corr, nan=0.0)

    # Volatility (rolling std of returns)
    vol_a = ret_a_series.rolling(window, min_periods=1).std().values
    vol_b = ret_b_series.rolling(window, min_periods=1).std().values

    # Spread momentum (rate of change of spread over window)
    spread_momentum = spread_series.diff(window).values
    spread_momentum = np.nan_to_num(spread_momentum, nan=0.0)

    # Spread mean-reversion signal: how far spread is from its rolling mean
    # Positive = spread above mean (expect reversion down), negative = below
    mean_reversion = spread - roll_mean

    # GNN strength (static, same for every timestep)
    gnn_feat = np.full_like(spread, gnn_strength)

    features = np.column_stack([
        spread,           
        spread_zscore,    
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


def split_data(features, prices_a, prices_b, dates, split_ratio=0.8):
    """ Chronologically split data into train and test sets."""

    n = len(features)
    split_idx = int(n * split_ratio)
    train = (features[:split_idx], prices_a[:split_idx], prices_b[:split_idx], dates[:split_idx])
    test = (features[split_idx:], prices_a[split_idx:], prices_b[split_idx:], dates[split_idx:])
    return train, test