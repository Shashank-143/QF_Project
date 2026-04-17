import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSITIVE_PAIRS_PATH = os.path.join(BASE_DIR, "..", "notebooks", "discovered_positive_pairs.csv")
NEGATIVE_PAIRS_PATH = os.path.join(BASE_DIR, "..", "notebooks", "discovered_negative_pairs.csv")
PRICE_DATA_PATH     = os.path.join(BASE_DIR, "..", "notebooks", "merged_data.csv")
RESULTS_DIR         = os.path.join(BASE_DIR, "results")

# ── Trading Parameters ────────────────────────────────────────────────────────
INITIAL_CAPITAL   = 10_000
TRANSACTION_COST  = 0.001
ACTION_THRESHOLD  = 0.1
TRAIN_TEST_SPLIT  = 0.8

# ── Network ───────────────────────────────────────────────────────────────────
HIDDEN_DIM        = 256

# ── PPO Core Hyperparameters ──────────────────────────────────────────────────
LEARNING_RATE = 1e-4   
GAMMA             = 0.99  
GAE_LAMBDA        = 0.95
CLIP_EPSILON  = 0.1     
VALUE_COEF        = 0.25      
ENTROPY_COEF = 0.001    
MAX_GRAD_NORM     = 0.5

# ── Rollout & Update Schedule ─────────────────────────────────────────────────
ROLLOUT_STEPS     = 512       
PPO_EPOCHS = 2      
MINI_BATCH_SIZE = 64         

# ── Training ──────────────────────────────────────────────────────────────────
NUM_EPISODES = 200   

# ── Environment ───────────────────────────────────────────────────────────────
LOOKBACK_WINDOW   = 20
MULTI_PAIR        = False    