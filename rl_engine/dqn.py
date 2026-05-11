"""Deep Q-Network (DQN) implemented from scratch in PyTorch."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from config import (
    GAMMA, EPSILON_START, EPSILON_MIN, EPSILON_DECAY,
    LEARNING_RATE, BATCH_SIZE, TARGET_UPDATE_FREQ, MODEL_DIR, REPLAY_BUFFER_SIZE,
    DQN_SEED,
)
from rl_engine.env import STATE_DIM, N_ACTIONS
from rl_engine.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)

_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class QNetwork(nn.Module):
    """MLP Q-network: state_dim → 256 → 128 → 64 → n_actions."""

    def __init__(self, state_dim: int = STATE_DIM, n_actions: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """DQN agent with experience replay, target network, and per-episode epsilon decay."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        n_actions: int = N_ACTIONS,
        lr: float = LEARNING_RATE,
        gamma: float = GAMMA,
        epsilon: float = EPSILON_START,
        epsilon_min: float = EPSILON_MIN,
        epsilon_decay: float = EPSILON_DECAY,
        batch_size: int = BATCH_SIZE,
        target_update_freq: int = TARGET_UPDATE_FREQ,
        replay_capacity: int = REPLAY_BUFFER_SIZE,
        seed: Optional[int] = None,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.episode_count = 0

        # Seed: explicit arg > config DQN_SEED > no seeding
        _seed = seed if seed is not None else DQN_SEED
        if _seed is not None:
            import random
            torch.manual_seed(_seed)
            np.random.seed(_seed)
            random.seed(_seed)

        self.q_network = QNetwork(state_dim, n_actions).to(_device)
        self.target_network = QNetwork(state_dim, n_actions).to(_device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.995)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(replay_capacity)
        self._opt_stepped = False  # track first optimizer.step() for scheduler ordering

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if evaluate or np.random.random() > self.epsilon:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(_device)
                q_values = self.q_network(state_t)
                return int(q_values.argmax(dim=1).item())
        return np.random.randint(self.n_actions)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> Optional[float]:
        """One step of Q-learning update. Returns loss value or None."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(_device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(_device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(_device)
        next_states_t = torch.FloatTensor(next_states).to(_device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(_device)

        # Current Q values
        current_q = self.q_network(states_t).gather(1, actions_t)

        # Double DQN: q_network selects action, target_network evaluates it
        with torch.no_grad():
            best_actions = self.q_network(next_states_t).argmax(dim=1, keepdim=True)
            max_next_q = self.target_network(next_states_t).gather(1, best_actions)
            target_q = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        self._opt_stepped = True

        return float(loss.item())

    def decay_epsilon(self):
        """Per-episode epsilon + LR decay (call after each full episode)."""
        self.episode_count += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self._opt_stepped:  # only step scheduler after at least one optimizer.step()
            self.scheduler.step()

        # Update target network per TARGET_UPDATE_FREQ episodes
        if self.episode_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, path: Optional[Path] = None):
        p = path or MODEL_DIR / "dqn_model.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epsilon": self.epsilon,
            "episode_count": self.episode_count,
        }, p)
        logger.info("Model saved to %s (ε=%.4f, ep=%d)", p, self.epsilon, self.episode_count)

    def load(self, path: Optional[Path] = None):
        p = path or MODEL_DIR / "dqn_model.pt"
        if not p.exists():
            logger.warning("Model file not found: %s", p)
            return
        checkpoint = torch.load(p, map_location=_device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self.episode_count = checkpoint.get("episode_count", 0)
        logger.info("Model loaded from %s", p)
