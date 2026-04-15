import numpy as np
import torch
import torch.optim as optim
import random

from config import (
    HIDDEN_DIM, LEARNING_RATE, GAMMA, TAU, ALPHA_INIT, AUTO_TUNE_ALPHA,
    BATCH_SIZE, BUFFER_CAPACITY, NUM_EPISODES, WARMUP_STEPS, UPDATE_EVERY,
    INITIAL_CAPITAL, TRANSACTION_COST, ACTION_THRESHOLD, LOOKBACK_WINDOW,
    TRAIN_TEST_SPLIT, MULTI_PAIR,
)
from data_loader import compute_pair_features, split_data
from environment import env_reset, env_step, get_observation, get_obs_dim
from networks import (
    create_actor, create_critic, clone_critic,
    sample_action, sample_action_and_logprob, actor_forward,
    critic_forward, compute_critic_loss, compute_actor_loss, soft_update,
)
from replay_buffer import create_buffer, buffer_push, buffer_sample, buffer_size


def prepare_pair_data(prices, pair, split_ratio):
    """Compute features and split into train/test for one pair."""
    stock_a, stock_b, pair_type, gnn_strength = pair
    features, pa, pb, dates = compute_pair_features(
        prices, stock_a, stock_b, gnn_strength, window=LOOKBACK_WINDOW
    )
    train, test = split_data(features, pa, pb, dates, split_ratio)
    return train, test, pair_type


def do_gradient_updates(actor, critic1, critic2, target_critic1, target_critic2,
                        actor_optim, critic1_optim, critic2_optim,
                        buffer, gamma, alpha, log_alpha, alpha_optim,
                        target_entropy, auto_tune, num_updates=1):
    """Perform SAC gradient updates. Returns updated alpha, critic_loss, actor_loss."""
    critic_loss_val = 0.0
    actor_loss_val = 0.0

    for _ in range(num_updates):
        batch = buffer_sample(buffer, BATCH_SIZE)

        # Update critics
        loss1, loss2 = compute_critic_loss(
            critic1, critic2, target_critic1, target_critic2,
            actor, batch, gamma, alpha
        )

        critic1_optim.zero_grad()
        loss1.backward()
        critic1_optim.step()

        critic2_optim.zero_grad()
        loss2.backward()
        critic2_optim.step()

        critic_loss_val = (loss1.item() + loss2.item()) / 2.0

        # Update actor
        a_loss, log_prob_mean = compute_actor_loss(
            actor, critic1, critic2, batch["obs"], alpha
        )

        actor_optim.zero_grad()
        a_loss.backward()
        actor_optim.step()

        actor_loss_val = a_loss.item()

        # Update alpha
        if auto_tune:
            alpha_loss = -(log_alpha * (log_prob_mean + target_entropy)).mean()
            alpha_optim.zero_grad()
            alpha_loss.backward()
            alpha_optim.step()
            alpha = log_alpha.exp().item()

        # Update target networks
        soft_update(target_critic1, critic1, TAU)
        soft_update(target_critic2, critic2, TAU)

    return alpha, critic_loss_val, actor_loss_val


def train_sac(pairs, price_data):
    """
    Train the SAC agent on all discovered pairs.

    Returns
    -------
    actor, history, pair_test_data
    """
    # Pre-compute features for all pairs
    pair_train_data = []
    pair_test_data = []
    for pair in pairs:
        try:
            train, test, pair_type = prepare_pair_data(price_data, pair, TRAIN_TEST_SPLIT)
            if len(train[0]) < LOOKBACK_WINDOW * 2:
                continue
            pair_train_data.append((train, pair_type, pair))
            pair_test_data.append((test, pair_type, pair))
        except KeyError:
            continue

    if not pair_train_data:
        raise ValueError("No valid pairs found in price data.")

    print(f"Prepared {len(pair_train_data)} pairs for training.")

    # Create networks
    obs_dim = get_obs_dim()
    action_dim = 1

    actor = create_actor(obs_dim, action_dim, HIDDEN_DIM)
    critic1 = create_critic(obs_dim, action_dim, HIDDEN_DIM)
    critic2 = create_critic(obs_dim, action_dim, HIDDEN_DIM)
    target_critic1 = clone_critic(critic1)
    target_critic2 = clone_critic(critic2)

    actor_optim = optim.Adam(actor["params"], lr=LEARNING_RATE)
    critic1_optim = optim.Adam(critic1["params"], lr=LEARNING_RATE)
    critic2_optim = optim.Adam(critic2["params"], lr=LEARNING_RATE)

    # Alpha (temperature)
    alpha = ALPHA_INIT
    log_alpha = None
    alpha_optim_obj = None
    target_entropy = -action_dim
    if AUTO_TUNE_ALPHA:
        log_alpha = torch.zeros(1, requires_grad=True)
        alpha_optim_obj = optim.Adam([log_alpha], lr=LEARNING_RATE)

    buffer = create_buffer(BUFFER_CAPACITY)

    history = {
        "episode_rewards": [],
        "episode_pnl": [],
        "critic_loss": [],
        "actor_loss": [],
        "alpha": [],
    }

    total_steps = 0

    for episode in range(NUM_EPISODES):
        # Pick pair(s) for this episode
        if MULTI_PAIR:
            selected = pair_train_data
        else:
            selected = [random.choice(pair_train_data)]

        ep_reward_total = 0.0
        ep_pnl_total = 0.0
        ep_critic_loss = 0.0
        ep_actor_loss = 0.0
        update_count = 0

        for (train_data, pair_type, pair_info) in selected:
            features, prices_a, prices_b, dates = train_data

            state = env_reset(prices_a, prices_b, features, pair_type,
                              INITIAL_CAPITAL, TRANSACTION_COST)
            obs = get_observation(state, 0)
            done = False

            while not done:
                # Select action
                if total_steps < WARMUP_STEPS:
                    action = random.uniform(-1.0, 1.0)
                else:
                    action = sample_action(actor, obs, deterministic=False)

                next_obs, reward, done, info = env_step(state, action, ACTION_THRESHOLD)
                buffer_push(buffer, obs, action, reward, next_obs, done)

                obs = next_obs
                ep_reward_total += reward
                total_steps += 1

                # Do gradient updates every UPDATE_EVERY steps
                if total_steps >= WARMUP_STEPS and buffer_size(buffer) >= BATCH_SIZE:
                    if total_steps % UPDATE_EVERY == 0:
                        alpha, cl, al = do_gradient_updates(
                            actor, critic1, critic2, target_critic1, target_critic2,
                            actor_optim, critic1_optim, critic2_optim,
                            buffer, GAMMA, alpha, log_alpha, alpha_optim_obj,
                            target_entropy, AUTO_TUNE_ALPHA, num_updates=1
                        )
                        ep_critic_loss += cl
                        ep_actor_loss += al
                        update_count += 1

            ep_pnl_total += state["total_pnl"]

        # Logging
        history["episode_rewards"].append(ep_reward_total)
        history["episode_pnl"].append(ep_pnl_total)
        history["critic_loss"].append(ep_critic_loss / max(update_count, 1))
        history["actor_loss"].append(ep_actor_loss / max(update_count, 1))
        history["alpha"].append(alpha)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(history["episode_rewards"][-10:])
            avg_pnl = np.mean(history["episode_pnl"][-10:])
            print(f"Episode {episode+1:4d}/{NUM_EPISODES} | "
                  f"Avg Reward: {avg_reward:8.4f} | "
                  f"Avg PnL: Rs.{avg_pnl:8.2f} | "
                  f"Alpha: {alpha:.4f} | "
                  f"Buffer: {buffer_size(buffer)}")

    print("Training complete.")
    return actor, history, pair_test_data
