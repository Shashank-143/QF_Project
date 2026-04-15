import numpy as np
import torch
from collections import deque
import random


def create_buffer(capacity=100000):
    """Create an empty replay buffer."""
    return {
        "storage": deque(maxlen=capacity),
        "capacity": capacity,
    }


def buffer_push(buffer, obs, action, reward, next_obs, done):
    """Store a transition in the buffer."""
    buffer["storage"].append((
        obs.copy() if isinstance(obs, np.ndarray) else obs,
        action,
        reward,
        next_obs.copy() if isinstance(next_obs, np.ndarray) else next_obs,
        float(done),
    ))


def buffer_sample(buffer, batch_size=256):
    """Sample a random mini-batch and return as tensors."""
    batch = random.sample(buffer["storage"], batch_size)
    obs, actions, rewards, next_obs, dones = zip(*batch)

    return {
        "obs": torch.FloatTensor(np.array(obs)),
        "action": torch.FloatTensor(np.array(actions)).unsqueeze(-1),
        "reward": torch.FloatTensor(np.array(rewards)).unsqueeze(-1),
        "next_obs": torch.FloatTensor(np.array(next_obs)),
        "done": torch.FloatTensor(np.array(dones)).unsqueeze(-1),
    }


def buffer_size(buffer):
    """Return current number of stored transitions."""
    return len(buffer["storage"])