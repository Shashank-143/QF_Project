"""
networks.py 
--------------------------
PPO uses a single Actor-Critic network with a shared backbone and two heads:
  • Actor head  → mean + log_std  (Gaussian policy, tanh-squashed)
  • Critic head → scalar V(s)     (state-value, NOT Q(s,a))

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -20
LOG_STD_MAX  =  2
EPSILON      = 1e-6


# ── Network construction ──────────────────────────────────────────────────────

def create_actor_critic(obs_dim: int, action_dim: int = 1, hidden_dim: int = 256):
    """
    Build a shared-backbone Actor-Critic network.
    """
    backbone = nn.Sequential(
        nn.Linear(obs_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
    )

    mean_head    = nn.Linear(hidden_dim, action_dim)
    log_std_head = nn.Linear(hidden_dim, action_dim)
    value_head   = nn.Linear(hidden_dim, 1)

    # Small init on output layers (standard PPO practice)
    for head in (mean_head, log_std_head, value_head):
        nn.init.orthogonal_(head.weight, gain=0.01)
        nn.init.zeros_(head.bias)

    all_params = (
        list(backbone.parameters())
        + list(mean_head.parameters())
        + list(log_std_head.parameters())
        + list(value_head.parameters())
    )

    return {
        "backbone":   backbone,
        "mean":       mean_head,
        "log_std":    log_std_head,
        "value_head": value_head,
        "params":     all_params,
    }


# ── Forward passes ────────────────────────────────────────────────────────────

def actor_critic_forward(net, obs):
    """Full forward pass → mean, log_std, value."""
    h       = net["backbone"](obs)
    mean    = net["mean"](h)
    log_std = torch.clamp(net["log_std"](h), LOG_STD_MIN, LOG_STD_MAX)
    value   = net["value_head"](h)
    return mean, log_std, value


def get_value(net, obs):
    """Critic-only forward — used for GAE bootstrapping at rollout end."""
    h = net["backbone"](obs)
    return net["value_head"](h)


# ── Action sampling ───────────────────────────────────────────────────────────

def sample_action(net, obs_np, deterministic: bool = False):
    obs  = torch.FloatTensor(obs_np).unsqueeze(0)
    mean, log_std, value = actor_critic_forward(net, obs)

    std = log_std.exp()
    dist = torch.distributions.Normal(mean, std)

    if deterministic:
        x_t = mean
    else:
        x_t = dist.rsample()

    action = torch.tanh(x_t)

    # log prob of RAW action (NOT tanh corrected)
    log_prob = dist.log_prob(x_t).sum(dim=-1, keepdim=True)

    return (
        action.item(),                     # env action
        log_prob.item(),
        value.item(),
        x_t.item()                        
    )


def sample_action_compat(net, obs_np, deterministic: bool = False):
    action, _, _ = sample_action(net, obs_np, deterministic=deterministic)
    return action


def evaluate_actions(net, obs, raw_action):
    mean, log_std, value = actor_critic_forward(net, obs)
    std  = log_std.exp()
    dist = torch.distributions.Normal(mean, std)

    log_prob = dist.log_prob(raw_action)
    log_prob = log_prob.sum(dim=-1, keepdim=True)

    entropy = dist.entropy().sum(dim=-1, keepdim=True)

    return log_prob, entropy, value

# ── PPO Loss ──────────────────────────────────────────────────────────────────

def compute_ppo_loss(net, batch, clip_eps, value_coef, entropy_coef):
    obs       = batch["obs"]
    actions = batch["raw_action"]
    old_logp  = batch["log_prob"]
    returns   = batch["returns"]
    adv       = batch["advantage"]

    new_logp, entropy, values = evaluate_actions(net, obs, actions)
    if torch.rand(1).item() < 0.02:
        print("\n--- PPO DEBUG ---")
        print("old_logp mean:", old_logp.mean().item())
        print("new_logp mean:", new_logp.mean().item())
        print("logp diff mean:", (new_logp - old_logp).mean().item())
        print("adv mean:", adv.mean().item(), "std:", adv.std().item())
        print("ratio mean:", torch.exp(new_logp - old_logp).mean().item())
        print("-----------------\n")

    # HARD clamp log prob difference before exp to prevent ratio explosion
    ratio = torch.exp(new_logp - old_logp)
    
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss   = 0.5 * (returns - values).pow(2).mean()
    entropy_loss = entropy.mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

    return loss, policy_loss.item(), value_loss.item(), entropy_loss.item()