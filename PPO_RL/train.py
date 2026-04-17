"""
train.py 
-----------------------

"""

import numpy as np
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "SAC_RL"))
from SAC_RL.data_loader import compute_pair_features, split_data
from PPO_RL.environment import env_reset, env_step, get_observation, get_obs_dim

from config import (
    INITIAL_CAPITAL, TRANSACTION_COST, ACTION_THRESHOLD,
    TRAIN_TEST_SPLIT, LOOKBACK_WINDOW, MULTI_PAIR, NUM_EPISODES,
)
from .ppo_agent import (
    create_ppo_agent, agent_act, agent_push,
    agent_buffer_full, agent_buffer_size, ppo_update,
)


def prepare_pair_data(prices, pair, split_ratio):
    """Compute features and split into train/test for one pair."""
    stock_a, stock_b, pair_type, gnn_strength = pair
    features, pa, pb, dates = compute_pair_features(
        prices, stock_a, stock_b, gnn_strength, window=LOOKBACK_WINDOW
    )
    train, test = split_data(features, pa, pb, dates, split_ratio)
    return train, test, pair_type


def train_ppo(pairs, price_data):
    """
    Train the PPO agent on all discovered pairs.

    Returns
    -------
    net          : trained network dict (backbone + actor + critic heads)
    history      : training metrics per episode
    pair_test_data : held-out test data for backtesting
    """
    # ── Pre-compute features ──────────────────────────────────────────────────
    pair_train_data = []
    pair_test_data  = []

    for pair in pairs:
        try:
            train, test, pair_type = prepare_pair_data(price_data, pair, TRAIN_TEST_SPLIT)
            if len(train[0]) < LOOKBACK_WINDOW * 2:
                continue
            pair_train_data.append((train, pair_type, pair))
            pair_test_data.append((test,  pair_type, pair))
        except KeyError:
            continue

    if not pair_train_data:
        raise ValueError("No valid pairs found in price data.")

    print(f"Prepared {len(pair_train_data)} pairs for training.")

    # ── Create agent ──────────────────────────────────────────────────────────
    obs_dim = get_obs_dim()
    agent   = create_ppo_agent(obs_dim, action_dim=1)

    history = {
        "episode_rewards": [],
        "episode_pnl":     [],
        "policy_loss":     [],
        "value_loss":      [],
        "entropy":         [],
    }

    # ── Episode loop ──────────────────────────────────────────────────────────
    for episode in range(NUM_EPISODES):

        n_pairs = min(3, len(pair_train_data))
        selected = random.sample(pair_train_data, n_pairs)

        ep_reward_total = 0.0
        ep_pnl_total    = 0.0
        ep_policy_loss  = 0.0
        ep_value_loss   = 0.0
        ep_entropy      = 0.0
        update_count    = 0

        for (train_data, pair_type, _pair_info) in selected:
            features, prices_a, prices_b, dates = train_data

            state = env_reset(prices_a, prices_b, features, pair_type,
                              INITIAL_CAPITAL, TRANSACTION_COST)
            obs   = get_observation(state, 0)
            done  = False

            while not done:
                # Collect one transition
                action, log_prob, value, raw_action = agent_act(agent, obs)
                next_obs, reward, done, _info = env_step(state, action, ACTION_THRESHOLD)
                agent_push(agent, obs, action, reward, done, log_prob, value, raw_action)

                obs              = next_obs
                ep_reward_total += reward

                # Fire PPO update when rollout buffer is full
                if agent_buffer_size(agent) >= 512:
                    logs = ppo_update(agent, obs)
                    ep_policy_loss += logs["policy_loss"]
                    ep_value_loss  += logs["value_loss"]
                    ep_entropy     += logs["entropy"]
                    update_count   += 1

            # Flush any remaining steps at episode end
            if agent_buffer_size(agent) > 0:
                logs = ppo_update(agent, obs)
                ep_policy_loss += logs["policy_loss"]
                ep_value_loss  += logs["value_loss"]
                ep_entropy     += logs["entropy"]
                update_count   += 1

            ep_pnl_total += state["total_pnl"]

        # ── Logging ───────────────────────────────────────────────────────────
        n = max(update_count, 1)
        history["episode_rewards"].append(ep_reward_total)
        history["episode_pnl"].append(ep_pnl_total)
        history["policy_loss"].append(ep_policy_loss / n)
        history["value_loss"].append(ep_value_loss   / n)
        history["entropy"].append(ep_entropy         / n)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(history["episode_rewards"][-10:])
            avg_pnl    = np.mean(history["episode_pnl"][-10:])
            print(
                f"Episode {episode+1:4d}/{NUM_EPISODES} | "
                f"Avg Reward: {avg_reward:8.4f} | "
                f"Avg PnL: Rs.{avg_pnl:8.2f} | "
                f"Policy Loss: {history['policy_loss'][-1]:.4f} | "
                f"Value Loss: {history['value_loss'][-1]:.4f} | "
                f"Entropy: {history['entropy'][-1]:.4f}"
            )

    print("Training complete.")
    return agent["net"], history, pair_test_data