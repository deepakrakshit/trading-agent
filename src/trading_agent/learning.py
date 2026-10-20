"""Small DQN/Double-DQN learner with portable, resumable research state.

PyTorch is loaded only when a trainer is constructed.  Replay, configuration,
and the other learning/evaluation modules can be used without it.  Training
operates on the Gymnasium protocol; the learner never receives an execution
adapter, ledger, broker client, credential, or model-registry handle.

Checkpoints contain both Q networks, Adam state, replay contents and sampling
state, exploration state, counters, configuration, and caller-provided research
metadata.  They do not capture an external environment/ledger: resume those
separately, or begin a new episode.  CPU continuation is deterministic for a
fixed PyTorch build, inputs and seed; bitwise cross-device equivalence is not
promised.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import importlib
import json
import math
from numbers import Integral
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Protocol

from .compatibility import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION as CHECKPOINT_SCHEMA_VERSION,
    CheckpointCompatibilityError,
    validate_checkpoint_compatibility,
)


class TorchDependencyError(ImportError):
    """Raised when training is requested without a usable PyTorch install."""


def _torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise TorchDependencyError("DQN training requires PyTorch; evaluation/replay do not") from exc


def _observation(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError("observations must be non-empty, finite, one-dimensional vectors")
    return result


@dataclass(frozen=True, slots=True)
class Transition:
    observation: tuple[float, ...]
    action: int
    reward: float
    next_observation: tuple[float, ...]
    terminated: bool = False
    truncated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation", _observation(self.observation))
        object.__setattr__(self, "next_observation", _observation(self.next_observation))
        if len(self.observation) != len(self.next_observation):
            raise ValueError("transition observation dimensions differ")
        if isinstance(self.action, bool) or not isinstance(self.action, Integral) or self.action < 0:
            raise ValueError("action must be a non-negative integer")
        if not math.isfinite(self.reward):
            raise ValueError("reward must be finite")
        # Gym/NumPy scalar values are normalised to safe checkpoint primitives.
        object.__setattr__(self, "action", int(self.action))
        object.__setattr__(self, "reward", float(self.reward))
        object.__setattr__(self, "terminated", bool(self.terminated))
        object.__setattr__(self, "truncated", bool(self.truncated))


class ReplayBuffer:
    """Bounded ring buffer that owns its sampler RNG and copies input vectors."""

    def __init__(self, capacity: int, *, seed: int = 0) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("replay capacity must be a positive integer")
        self.capacity = capacity
        self._items: list[Transition] = []
        self._position = 0
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    def append(self, transition: Transition) -> None:
        if not isinstance(transition, Transition):
            raise TypeError("replay accepts Transition instances")
        if len(self._items) < self.capacity:
            self._items.append(transition)
        else:
            self._items[self._position] = transition
        self._position = (self._position + 1) % self.capacity

    def sample(self, batch_size: int) -> tuple[Transition, ...]:
        if not 0 < batch_size <= len(self):
            raise ValueError("batch_size must be positive and no larger than replay")
        return tuple(self._rng.sample(self._items, batch_size))

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "items": [asdict(item) for item in self._items],
            "position": self._position,
            "rng_state": self._rng.getstate(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> ReplayBuffer:
        result = cls(int(state["capacity"]))
        items = [Transition(**item) for item in state["items"]]
        position = int(state["position"])
        if len(items) > result.capacity or not 0 <= position < result.capacity:
            raise ValueError("invalid replay checkpoint shape")
        if len(items) < result.capacity and position != len(items):
            raise ValueError("invalid replay checkpoint write position")
        result._items = items
        result._position = position
        result._rng.setstate(state["rng_state"])
        return result


@dataclass(frozen=True, slots=True)
class DQNConfig:
    observation_dim: int
    n_actions: int
    hidden_sizes: tuple[int, ...] = (64, 64)
    gamma: float = 0.99
    learning_rate: float = 1e-3
    replay_capacity: int = 10_000
    batch_size: int = 32
    learning_starts: int = 32
    target_update_interval: int = 100
    train_frequency: int = 1
    double_dqn: bool = True
    bootstrap_truncated: bool = True
    gradient_clip: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 1_000
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_sizes", tuple(self.hidden_sizes))
        positive_integers = (
            self.observation_dim, self.n_actions, self.replay_capacity,
            self.batch_size, self.target_update_interval, self.train_frequency,
            self.epsilon_decay_steps, *self.hidden_sizes,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in positive_integers):
            raise ValueError("DQN dimensions/capacities/intervals must be positive integers")
        if not isinstance(self.learning_starts, int) or self.learning_starts < 0:
            raise ValueError("learning_starts cannot be negative")
        if self.batch_size > self.replay_capacity:
            raise ValueError("batch_size exceeds replay capacity")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be between zero and one")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be finite and positive")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("epsilon schedule must satisfy 0 <= end <= start <= 1")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("device must explicitly select cpu or cuda")


class TrainingEnvironment(Protocol):
    def reset(self, *, seed: int | None = None) -> tuple[Sequence[float], Mapping[str, Any]]:
        ...

    def step(self, action: int) -> tuple[Sequence[float], float, bool, bool, Mapping[str, Any]]:
        ...


class DQNTrainer:
    """Compact MLP DQN; Double-DQN is the default target estimator."""

    CHECKPOINT_VERSION = CHECKPOINT_SCHEMA_VERSION

    def __init__(self, config: DQNConfig, *, metadata: Mapping[str, Any] | None = None) -> None:
        self.config = config
        self.torch = _torch()
        torch = self.torch
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable; select device='cpu'")
        self.device = torch.device(config.device)
        # No dropout/noisy layers: network construction is the only torch RNG
        # consumer.  Preserve the application's global RNG while seeding it.
        with torch.random.fork_rng(devices=[]):
            # Layers are initialised on CPU before moving to the chosen
            # device, so no application CUDA generator needs to be reseeded.
            torch.random.default_generator.manual_seed(config.seed)
            layers = []
            previous = config.observation_dim
            for size in config.hidden_sizes:
                layers.extend((torch.nn.Linear(previous, size), torch.nn.ReLU()))
                previous = size
            layers.append(torch.nn.Linear(previous, config.n_actions))
            self.online = torch.nn.Sequential(*layers).to(self.device)
        import copy

        self.target = copy.deepcopy(self.online)
        self.target.eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay = ReplayBuffer(config.replay_capacity, seed=config.seed + 1)
        self._rng = random.Random(config.seed)
        self.environment_steps = 0
        self.gradient_steps = 0
        self.episodes = 0
        # Metadata should identify feature/action schemas, data, reward, costs,
        # execution and code versions.  JSON round-trip prevents opaque objects
        # from requiring unsafe pickle deserialisation.
        self.metadata = json.loads(json.dumps(dict(metadata or {}), allow_nan=False, sort_keys=True))

    def _validate_observation(self, observation: Sequence[float]) -> tuple[float, ...]:
        result = _observation(observation)
        if len(result) != self.config.observation_dim:
            raise ValueError("observation dimension does not match DQNConfig")
        return result

    @property
    def epsilon(self) -> float:
        fraction = min(1.0, self.environment_steps / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + fraction * (self.config.epsilon_end - self.config.epsilon_start)

    def select_action(self, observation: Sequence[float], *, epsilon: float = 0.0) -> int:
        values = self._validate_observation(observation)
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between zero and one")
        if epsilon > 0 and self._rng.random() < epsilon:
            return self._rng.randrange(self.config.n_actions)
        with self.torch.no_grad():
            tensor = self.torch.tensor(values, dtype=self.torch.float32, device=self.device).unsqueeze(0)
            return int(self.online(tensor).argmax(dim=1).item())

    def act(self, observation: Sequence[float]) -> int:
        """Deterministic evaluation/paper action: exploration is disabled."""

        return self.select_action(observation, epsilon=0.0)

    __call__ = act

    def observe(self, transition: Transition) -> float | None:
        self._validate_observation(transition.observation)
        self._validate_observation(transition.next_observation)
        if not 0 <= transition.action < self.config.n_actions:
            raise ValueError("transition action is outside the configured action space")
        self.replay.append(transition)
        self.environment_steps += 1
        if self.environment_steps >= self.config.learning_starts and self.environment_steps % self.config.train_frequency == 0:
            return self.train_step()
        return None

    def _targets(self, rewards: Any, next_observations: Any, terminals: Any) -> Any:
        torch = self.torch
        with torch.no_grad():
            if self.config.double_dqn:
                actions = self.online(next_observations).argmax(dim=1, keepdim=True)
                continuation = self.target(next_observations).gather(1, actions).squeeze(1)
            else:
                continuation = self.target(next_observations).max(dim=1).values
            return rewards + self.config.gamma * (1.0 - terminals) * continuation

    def train_step(self) -> float | None:
        if len(self.replay) < self.config.batch_size:
            return None
        torch = self.torch
        batch = self.replay.sample(self.config.batch_size)
        observations = torch.tensor([row.observation for row in batch], dtype=torch.float32, device=self.device)
        next_observations = torch.tensor([row.next_observation for row in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([row.action for row in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor([row.reward for row in batch], dtype=torch.float32, device=self.device)
        terminals = torch.tensor(
            [row.terminated or (row.truncated and not self.config.bootstrap_truncated) for row in batch],
            dtype=torch.float32, device=self.device,
        )
        predicted = self.online(observations).gather(1, actions).squeeze(1)
        expected = self._targets(rewards, next_observations, terminals)
        loss = torch.nn.functional.smooth_l1_loss(predicted, expected)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite DQN loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip, error_if_nonfinite=True)
        self.optimizer.step()
        self.gradient_steps += 1
        if self.gradient_steps % self.config.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def train(self, env: TrainingEnvironment, total_steps: int, *, epsilon: float | None = None) -> dict[str, Any]:
        """Collect transitions and optimise on a caller-supplied paper env.

        Each call starts a fresh episode.  A budget ending mid-episode is a
        truncation (and bootstraps by default), never a false terminal state.
        """

        if not isinstance(total_steps, int) or total_steps <= 0:
            raise ValueError("total_steps must be a positive integer")
        observation, _ = env.reset(seed=self.config.seed + self.episodes)
        losses = []
        start_updates = self.gradient_steps
        for index in range(total_steps):
            action = self.select_action(observation, epsilon=self.epsilon if epsilon is None else epsilon)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            budget_end = index + 1 == total_steps and not terminated
            loss = self.observe(Transition(
                tuple(observation), action, float(reward), tuple(next_observation),
                bool(terminated), bool(truncated or budget_end),
            ))
            if loss is not None:
                losses.append(loss)
            observation = next_observation
            if terminated or truncated or budget_end:
                self.episodes += 1
                if index + 1 < total_steps:
                    observation, _ = env.reset(seed=self.config.seed + self.episodes)
        return {"steps": total_steps, "updates": self.gradient_steps - start_updates, "losses": tuple(losses)}

    def save_checkpoint(self, path: str | Path) -> Path:
        path = Path(path)
        if not path.parent.is_dir():
            raise FileNotFoundError(f"checkpoint directory does not exist: {path.parent}")
        payload = {
            "format": CHECKPOINT_FORMAT, "version": self.CHECKPOINT_VERSION,
            "config": asdict(self.config), "metadata": self.metadata,
            "torch_version": str(self.torch.__version__),
            "online": self.online.state_dict(), "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(), "replay": self.replay.state_dict(),
            "exploration_rng": self._rng.getstate(), "environment_steps": self.environment_steps,
            "gradient_steps": self.gradient_steps, "episodes": self.episodes,
        }
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
                temporary = handle.name
                self.torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
        return path

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
        expected_compatibility: Mapping[str, Any] | None = None,
    ) -> DQNTrainer:
        torch = _torch()
        state = torch.load(Path(path), map_location=device, weights_only=True)
        if state.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("unsupported DQN checkpoint format")
        if state.get("version") != cls.CHECKPOINT_VERSION:
            raise ValueError(
                "unsupported DQN checkpoint version; legacy checkpoints require explicit migration/retraining"
            )
        if not isinstance(state.get("metadata"), Mapping):
            raise ValueError("DQN checkpoint metadata is missing or invalid")
        if expected_compatibility is not None:
            try:
                validate_checkpoint_compatibility(
                    state["metadata"], expected_compatibility, context=str(path)
                )
            except CheckpointCompatibilityError:
                raise
        config = dict(state["config"])
        config["device"] = device
        result = cls(DQNConfig(**config), metadata=state["metadata"])
        result.online.load_state_dict(state["online"])
        result.target.load_state_dict(state["target"])
        result.optimizer.load_state_dict(state["optimizer"])
        result.replay = ReplayBuffer.from_state_dict(state["replay"])
        if result.replay.capacity != result.config.replay_capacity:
            raise ValueError("checkpoint replay capacity does not match configuration")
        result._rng.setstate(state["exploration_rng"])
        result.environment_steps = int(state["environment_steps"])
        result.gradient_steps = int(state["gradient_steps"])
        result.episodes = int(state["episodes"])
        return result


__all__ = ["DQNConfig", "DQNTrainer", "ReplayBuffer", "TorchDependencyError", "TrainingEnvironment", "Transition"]
