import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSITIVE_PAIRS_PATH = os.path.join(BASE_DIR, "..", "notebooks", "discovered_positive_pairs.csv")
NEGATIVE_PAIRS_PATH = os.path.join(BASE_DIR, "..", "notebooks", "discovered_negative_pairs.csv")
PRICE_DATA_PATH = os.path.join(BASE_DIR, "..", "notebooks", "merged_data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Trading Parameters 
INITIAL_CAPITAL = 10000          
TRANSACTION_COST = 0.001          
ACTION_THRESHOLD = 0.2            # |action| below this -> hold / close position
TRAIN_TEST_SPLIT = 0.8            

# SAC Hyperparameters 
HIDDEN_DIM = 256                  # Hidden layer size for actor & critic
LEARNING_RATE = 3e-4              # Adam learning rate
GAMMA = 0.99                      # Discount factor
TAU = 0.005                       # Polyak averaging coefficient
ALPHA_INIT = 0.2                  # Initial entropy temperature
AUTO_TUNE_ALPHA = True            # Auto-tune α during training
BATCH_SIZE = 256                  # Mini-batch size for updates
BUFFER_CAPACITY = 10000         # Replay buffer max transitions
NUM_EPISODES = 100                # Total training episodes (each does ~1975 updates)
WARMUP_STEPS = 500                # Random actions before learning starts
UPDATE_EVERY = 4                  # Env steps between gradient updates

LOOKBACK_WINDOW = 20              
MULTI_PAIR = False                # True  = trade multiple pairs simultaneously