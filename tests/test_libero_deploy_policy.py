import pytest

from experiments.libero.w0_policy import deploy_policy
from experiments.libero.w0_policy.deploy_policy import LiberoPosttrainPolicy


class _FakeModel:
    def __init__(self):
        self._inference_state = {"next_chunk_index": 1}
        self.reset_calls = 0

    def reset_history(self):
        self.reset_calls += 1


def test_new_observation_clears_video_history_and_chunk_state():
    policy = LiberoPosttrainPolicy.__new__(LiberoPosttrainPolicy)
    policy.model = _FakeModel()
    policy._episode_prefilled = True
    policy._chunks_since_video_prefill = 1

    policy._reset_for_new_observation()

    assert policy.model.reset_calls == 1
    assert policy.model._inference_state is None
    assert policy._episode_prefilled is False
    assert policy._chunks_since_video_prefill == 0


def test_checkpoint_chunk_size_is_read_before_model_construction(monkeypatch):
    monkeypatch.setattr(
        deploy_policy.torch,
        "load",
        lambda *args, **kwargs: {"behavior_config": {"action_chunk_size": 8}},
    )
    assert deploy_policy._checkpoint_action_chunk_size("unused.pt") == 8


def test_checkpoint_chunk_size_requires_embedded_contract(monkeypatch):
    monkeypatch.setattr(deploy_policy.torch, "load", lambda *args, **kwargs: {})
    with pytest.raises(ValueError, match="no embedded action_chunk_size"):
        deploy_policy._checkpoint_action_chunk_size("unused.pt")
