import torch
import torch.optim as optim
import numpy as np

from .networks import create_actor_critic, sample_action, get_value, compute_ppo_loss
from .rollout_buffer import (
    create_rollout_buffer, rollout_push, rollout_full, rollout_size,
    compute_gae, rollout_get_batches, rollout_clear,
)
from .config import (
    HIDDEN_DIM, LEARNING_RATE, GAMMA, GAE_LAMBDA,
    CLIP_EPSILON, VALUE_COEF, ENTROPY_COEF, MAX_GRAD_NORM,
    ROLLOUT_STEPS, PPO_EPOCHS, MINI_BATCH_SIZE,
)


def create_ppo_agent(obs_dim: int, action_dim: int = 1):
    net       = create_actor_critic(obs_dim, action_dim, HIDDEN_DIM)
    optimizer = optim.Adam(net["params"], lr=LEARNING_RATE, eps=1e-5)
    buffer    = create_rollout_buffer(ROLLOUT_STEPS, obs_dim)

    return {
        "net":        net,
        "optimizer":  optimizer,
        "buffer":     buffer,
        "obs_dim":    obs_dim,
        "action_dim": action_dim,
    }


def agent_act(agent, obs_np, deterministic: bool = False):
    action, log_prob, value, raw_action = sample_action(
        agent["net"], obs_np, deterministic=deterministic
    )
    return action, log_prob, value, raw_action


def agent_push(agent, obs, action, reward, done, log_prob, value, raw_action):
    rollout_push(
        agent["buffer"], obs, action, reward, done, log_prob, value, raw_action
    )


def agent_buffer_full(agent) -> bool:
    return rollout_full(agent["buffer"])


def agent_buffer_size(agent) -> int:
    return rollout_size(agent["buffer"])


def ppo_update(agent, last_obs_np) -> dict:
    net    = agent["net"]
    buf    = agent["buffer"]
    optim_ = agent["optimizer"]

    # Bootstrap final value
    last_obs   = torch.FloatTensor(last_obs_np).unsqueeze(0)
    last_value = get_value(net, last_obs).item()

    compute_gae(buf, last_value, GAMMA, GAE_LAMBDA)

    total_policy_loss = 0.0
    total_value_loss  = 0.0
    total_entropy     = 0.0
    n_updates         = 0

    for _ in range(PPO_EPOCHS):
        for batch in rollout_get_batches(buf, MINI_BATCH_SIZE):

            loss, pl, vl, ent = compute_ppo_loss(
                net, batch, CLIP_EPSILON, VALUE_COEF, ENTROPY_COEF
            )

            optim_.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net["params"], MAX_GRAD_NORM)
            optim_.step()

            total_policy_loss += float(pl)
            total_value_loss  += float(vl)
            total_entropy     += float(ent)
            n_updates         += 1

    rollout_clear(buf)

    return {
        "policy_loss": total_policy_loss / max(n_updates, 1),
        "value_loss":  total_value_loss  / max(n_updates, 1),
        "entropy":     total_entropy     / max(n_updates, 1),
    }