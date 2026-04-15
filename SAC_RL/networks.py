import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPSILON = 1e-6

def create_actor(obs_dim, action_dim=1, hidden_dim=256):
    """Create a Gaussian policy actor network."""

    net = nn.Sequential(
        nn.Linear(obs_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
    )
    mean_head = nn.Linear(hidden_dim, action_dim)
    log_std_head = nn.Linear(hidden_dim, action_dim)

    # Initialise output heads with small weights
    nn.init.uniform_(mean_head.weight, -3e-3, 3e-3)
    nn.init.uniform_(mean_head.bias, -3e-3, 3e-3)
    nn.init.uniform_(log_std_head.weight, -3e-3, 3e-3)
    nn.init.uniform_(log_std_head.bias, -3e-3, 3e-3)

    params = list(net.parameters()) + list(mean_head.parameters()) + list(log_std_head.parameters())

    return {
        "net": net,
        "mean": mean_head,
        "log_std": log_std_head,
        "params": params,
    }


def create_critic(obs_dim, action_dim=1, hidden_dim=256):
    """Create a Q-value critic network: Q(s, a) → scalar."""

    net = nn.Sequential(
        nn.Linear(obs_dim + action_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )
    return {"net": net, "params": list(net.parameters())}


def clone_critic(critic):
    """ Deep-copy a critic network (for target networks)."""

    net_copy = copy.deepcopy(critic["net"])
    return {"net": net_copy, "params": list(net_copy.parameters())}


def actor_forward(actor, obs):
    """ Forward pass through the actor."""

    h = actor["net"](obs)
    mean = actor["mean"](h)
    log_std = actor["log_std"](h)
    log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
    return mean, log_std


def critic_forward(critic, obs, action):
    """ Forward pass through a critic: Q(s, a)."""

    x = torch.cat([obs, action], dim=-1)
    return critic["net"](x)


# Action sampling
def sample_action(actor, obs_np, deterministic=False):
    """ Sample an action from the actor given a single observation."""

    obs = torch.FloatTensor(obs_np).unsqueeze(0)
    mean, log_std = actor_forward(actor, obs)

    if deterministic:
        action = torch.tanh(mean)
    else:
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()        # reparameterisation trick
        action = torch.tanh(x_t)

    return action.squeeze().detach().numpy().item()


def sample_action_and_logprob(actor, obs):
    """ Sample action + compute log-probability (for training)."""
    mean, log_std = actor_forward(actor, obs)
    std = log_std.exp()
    normal = torch.distributions.Normal(mean, std)
    x_t = normal.rsample()
    action = torch.tanh(x_t)

    # Log-probability with tanh correction
    log_prob = normal.log_prob(x_t)
    log_prob -= torch.log(1 - action.pow(2) + EPSILON)
    log_prob = log_prob.sum(dim=-1, keepdim=True)

    return action, log_prob


# Loss computation
def compute_critic_loss(critic1, critic2, target_critic1, target_critic2,
                        actor, batch, gamma, alpha):
    """ Compute SAC twin-critic loss."""

    obs = batch["obs"]
    action = batch["action"]
    reward = batch["reward"]
    next_obs = batch["next_obs"]
    done = batch["done"]

    with torch.no_grad():
        next_action, next_log_prob = sample_action_and_logprob(actor, next_obs)
        target_q1 = critic_forward(target_critic1, next_obs, next_action)
        target_q2 = critic_forward(target_critic2, next_obs, next_action)
        target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
        target_value = reward + (1.0 - done) * gamma * target_q

    q1 = critic_forward(critic1, obs, action)
    q2 = critic_forward(critic2, obs, action)
    loss1 = F.mse_loss(q1, target_value)
    loss2 = F.mse_loss(q2, target_value)

    return loss1, loss2


def compute_actor_loss(actor, critic1, critic2, obs, alpha):
    """Compute SAC actor loss (policy gradient with entropy regularisation)."""

    action, log_prob = sample_action_and_logprob(actor, obs)
    q1 = critic_forward(critic1, obs, action)
    q2 = critic_forward(critic2, obs, action)
    min_q = torch.min(q1, q2)

    loss = (alpha * log_prob - min_q).mean()
    return loss, log_prob.detach().mean()


# Target network update
def soft_update(target_critic, source_critic, tau):
    """Polyak-average target network parameters toward source."""
    for tp, sp in zip(target_critic["params"], source_critic["params"]):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)