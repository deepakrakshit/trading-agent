"""Trainer/replay checks exercise exact resumed CPU updates, not only loading."""

from dataclasses import replace
import random

import pytest

from trading_agent.learning import DQNConfig, DQNTrainer, ReplayBuffer, Transition
from trading_agent.compatibility import CheckpointCompatibilityError


def transition(index):
    return Transition((float(index) / 100, 1.0), index % 2, index / 1_000, ((index + 1) / 100, 1.0), index % 7 == 0)


def test_replay_ring_and_sampler_restore_copy_inputs():
    buffer = ReplayBuffer(3, seed=11)
    values = [1.0, 2.0]
    item = Transition(values, 0, 0.1, values)
    buffer.append(item)
    values[0] = 99
    assert buffer.sample(1)[0].observation == (1, 2)
    for index in range(7):
        buffer.append(transition(index))
    assert len(buffer) == 3
    restored = ReplayBuffer.from_state_dict(buffer.state_dict())
    assert buffer.sample(3) == restored.sample(3)
    buffer.append(transition(99))
    restored.append(transition(99))
    assert buffer.state_dict() == restored.state_dict()
    with pytest.raises(ValueError):
        buffer.sample(4)


def test_numpy_gym_scalars_are_safe_checkpoint_values(tmp_path):
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    trainer = DQNTrainer(config())
    item = Transition(numpy.array([0, 1]), numpy.int64(1), numpy.float32(0.1), numpy.array([1, 1]), numpy.bool_(False), numpy.bool_(True))
    assert type(item.action) is int and type(item.reward) is float
    assert type(item.terminated) is bool and type(item.truncated) is bool
    trainer.observe(item)
    restored = DQNTrainer.load_checkpoint(trainer.save_checkpoint(tmp_path / "numpy-scalars.pt"))
    assert restored.replay.sample(1) == (item,)


@pytest.mark.parametrize("kwargs", [{"gamma": float("nan")}, {"batch_size": 11, "replay_capacity": 10}, {"n_actions": 0}])
def test_invalid_config(kwargs):
    values = {"observation_dim": 2, "n_actions": 2}
    values.update(kwargs)
    with pytest.raises(ValueError):
        DQNConfig(**values)


def config():
    return DQNConfig(2, 2, hidden_sizes=(8,), batch_size=4, replay_capacity=32, learning_starts=4, target_update_interval=3, seed=37)


def test_checkpoint_restores_weights_optimizer_replay_rng_and_next_updates(tmp_path):
    torch = pytest.importorskip("torch")
    trainer = DQNTrainer(config(), metadata={"data_version": "SYNTHETIC-test", "feature_schema": ["x", "cash"], "reward": "log_equity-v1"})
    initial = [value.detach().clone() for value in trainer.online.parameters()]
    for index in range(15):
        trainer.select_action((0.1, 1), epsilon=0.75)
        trainer.observe(transition(index))
    assert any(not torch.equal(before, after) for before, after in zip(initial, trainer.online.parameters()))
    path = trainer.save_checkpoint(tmp_path / "candidate.pt")
    restored = DQNTrainer.load_checkpoint(path)
    assert restored.config == trainer.config and restored.metadata == trainer.metadata
    assert restored.environment_steps == trainer.environment_steps
    assert restored.gradient_steps == trainer.gradient_steps
    assert restored.replay.state_dict() == trainer.replay.state_dict()
    assert restored.act((0.3, 1)) == trainer.act((0.3, 1))
    for index in range(15, 25):
        assert restored.select_action((0.2, 1), epsilon=0.8) == trainer.select_action((0.2, 1), epsilon=0.8)
        assert restored.observe(transition(index)) == trainer.observe(transition(index))
    for key, value in trainer.online.state_dict().items():
        assert torch.equal(value, restored.online.state_dict()[key])
    for key, value in trainer.target.state_dict().items():
        assert torch.equal(value, restored.target.state_dict()[key])


def test_checkpoint_semantic_compatibility_requires_schema_actions_and_versions(tmp_path):
    pytest.importorskip("torch")
    compatibility = {
        "compatibility_version": "paper-compat-v1",
        "config_version": "research-v1",
        "observation_dim": 2,
        "n_actions": 2,
        "feature_schema": ["x", "cash"],
        "feature_version": "features-v1",
        "target_exposures": [0.0, 1.0],
        "reward_version": "log-post-cost-equity-v1",
        "execution_version": "causal-open-v2",
        "bar_interval_seconds": 300,
        "cost_model_version": "cost-v1",
        "risk_version": "risk-v1",
        "environment_config_hash": "env-hash",
        "instrument": "ABC",
        "policy_id": "candidate-v1",
    }
    trainer = DQNTrainer(config(), metadata={"checkpoint_compatibility": compatibility})
    path = trainer.save_checkpoint(tmp_path / "semantic.pt")
    assert DQNTrainer.load_checkpoint(path, expected_compatibility=compatibility).config == trainer.config
    for field, value in (("feature_schema", ["cash", "x"]), ("target_exposures", [0.0, 0.5]), ("execution_version", "other")):
        incompatible = dict(compatibility)
        incompatible[field] = value
        with pytest.raises(CheckpointCompatibilityError, match="compatibility mismatch"):
            DQNTrainer.load_checkpoint(path, expected_compatibility=incompatible)
    bare = DQNTrainer(config())
    bare_path = bare.save_checkpoint(tmp_path / "bare.pt")
    with pytest.raises(CheckpointCompatibilityError, match="missing required checkpoint_compatibility"):
        DQNTrainer.load_checkpoint(bare_path, expected_compatibility=compatibility)


def test_same_seed_independent_trainers_and_global_rng_isolation():
    torch = pytest.importorskip("torch")
    python_state = random.getstate()
    torch_state = torch.random.get_rng_state().clone()
    first, second = DQNTrainer(config()), DQNTrainer(config())
    assert python_state == random.getstate()
    assert torch.equal(torch_state, torch.random.get_rng_state())
    for index in range(10):
        assert first.observe(transition(index)) == second.observe(transition(index))
    assert all(torch.equal(a, b) for a, b in zip(first.online.parameters(), second.online.parameters()))


def test_double_dqn_uses_online_argmax_but_target_value_and_terminal_mask():
    torch = pytest.importorskip("torch")
    base = replace(config(), hidden_sizes=(), gamma=0.5)
    double = DQNTrainer(base)
    standard = DQNTrainer(replace(base, double_dqn=False))
    for trainer in (double, standard):
        with torch.no_grad():
            trainer.online[0].weight.zero_()
            trainer.online[0].bias.copy_(torch.tensor([2.0, 1.0]))
            trainer.target[0].weight.zero_()
            trainer.target[0].bias.copy_(torch.tensor([3.0, 9.0]))
    rewards = torch.tensor([1.0, 1.0])
    observations = torch.zeros((2, 2))
    terminals = torch.tensor([0.0, 1.0])
    assert double._targets(rewards, observations, terminals).tolist() == [2.5, 1]
    assert standard._targets(rewards, observations, terminals).tolist() == [5.5, 1]


def test_time_limit_truncation_bootstraps_but_true_terminal_does_not():
    torch = pytest.importorskip("torch")
    settings = replace(config(), hidden_sizes=(), gamma=0.5, batch_size=1, learning_starts=1)
    losses = []
    for terminated, truncated, bootstrap in ((False, True, True), (False, True, False), (True, False, True)):
        trainer = DQNTrainer(replace(settings, bootstrap_truncated=bootstrap))
        with torch.no_grad():
            trainer.online[0].weight.zero_()
            trainer.online[0].bias.zero_()
            trainer.target[0].weight.zero_()
            trainer.target[0].bias.fill_(2)
        losses.append(trainer.observe(Transition((0, 1), 0, 0, (0, 1), terminated, truncated)))
    assert losses == [0.5, 0, 0]


def test_gym_training_loop_and_truncated_budget():
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    from trading_agent.env import MarketBar, TradingEnv

    bars = [MarketBar(index, 100, 101, 99, 100 + (index % 2), 10) for index in range(8)]
    env = TradingEnv(bars, target_exposures=(0, 1))
    settings = replace(config(), observation_dim=env.observation_space.shape[0])
    first, second = DQNTrainer(settings), DQNTrainer(settings)
    report = first.train(env, 12)
    assert report == second.train(TradingEnv(bars, target_exposures=(0, 1)), 12)
    assert report["updates"] == 9
    assert first.environment_steps == 12
    assert first.replay.state_dict()["items"][-1]["truncated"]
    assert not first.replay.state_dict()["items"][-1]["terminated"]


def test_rejects_nonfinite_transition_and_bad_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="finite"):
        Transition((0, 1), 0, float("nan"), (0, 1))
    torch = pytest.importorskip("torch")
    path = tmp_path / "wrong.pt"
    torch.save({"format": "other", "version": 1}, path)
    with pytest.raises(ValueError, match="format"):
        DQNTrainer.load_checkpoint(path)
    legacy = tmp_path / "legacy.pt"
    torch.save({"format": "trading-agent-dqn", "version": 1}, legacy)
    with pytest.raises(ValueError, match="legacy checkpoints"):
        DQNTrainer.load_checkpoint(legacy)


def test_offline_train_restore_evaluate_and_stage_without_safety_receipts(tmp_path):
    """The real paper components compose without any provider/network service."""
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    from datetime import datetime, timedelta, timezone
    from trading_agent.env import MarketBar, SimplePaperExecution, TradingEnv
    from trading_agent.evaluation import evaluate_baselines, evaluate_policy
    from trading_agent.registry import CandidateRegistry, EvaluationSummary, EvaluationWindow, PromotionEvidence

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bars = [MarketBar(start + timedelta(days=index), 100, 102, 99, 100 + index % 2, 100) for index in range(30)]

    def factory(rows):
        return TradingEnv(rows, target_exposures=(0, 1), execution=SimplePaperExecution(0.001, 5))

    trainer = DQNTrainer(replace(config(), observation_dim=9), metadata={"data_version": "SYNTHETIC-smoke", "action_schema": [0, 1], "feature_schema": "default-causal-v1", "reward": "log-equity-v1", "costs": {"fee_rate": 0.001, "slippage_bps": 5}})
    trainer.train(factory(bars[:10]), 18)
    checkpoint = trainer.save_checkpoint(tmp_path / "candidate.pt")
    policy = DQNTrainer.load_checkpoint(checkpoint)
    candidate = evaluate_policy(factory(bars[10:]), policy)
    baselines = evaluate_baselines(lambda: factory(bars[10:]))
    assert candidate == evaluate_policy(factory(bars[10:]), trainer)
    bounds = ((10, 20), (20, 30))

    def report(policy_id, result, policy_callable):
        windows = tuple(EvaluationWindow(first, last, evaluate_policy(factory(bars[first:last]), policy_callable).metrics.net_return, "SYNTHETIC-development") for first, last in bounds)
        return EvaluationSummary(policy_id, "SYNTHETIC-smoke:unseen10-30:fees0.001:slip5", result.metrics, windows, (0,))

    # Empty safety receipts intentionally cannot become a promotion just
    # because a checkpoint trains, reloads and obtains an evaluation score.
    evidence = PromotionEvidence("integration-trial", report("candidate", candidate, policy), accounting_ok=True, data_integrity_ok=True, risk_ok=True)
    with CandidateRegistry(tmp_path / "registry.sqlite") as registry:
        registry.register("candidate", checkpoint, metadata=policy.metadata)
        decision = registry.consider("candidate", evidence)
        assert not decision.accepted
        assert "shadow_failed" in decision.reasons and "recovery_failed" in decision.reasons
        assert registry.active is None
        assert registry.history()[-1]["kind"] == "rejected"
    assert set(baselines) == {"cash", "rule", "random"}
