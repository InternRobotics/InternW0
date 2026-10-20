import numpy as np
import pytest

from w0.datasets.libero_posttrain import frame_counts, full_window_counts, locate_global_window


def test_frame_counts_match_lerobot_frame_indexing():
    episodes = [{"length": 80}, {"length": 100}, {"length": 40}]
    counts = frame_counts(episodes)
    np.testing.assert_array_equal(counts, [80, 100, 40])


def test_full_window_counts_weight_long_episodes_by_available_windows():
    episodes = [{"length": 80}, {"length": 100}, {"length": 40}]
    counts = full_window_counts(episodes, action_horizon=64)
    np.testing.assert_array_equal(counts, [17, 37, 1])


def test_locate_global_window_maps_boundaries_without_episode_bias():
    cumulative = np.cumsum([2, 3, 1], dtype=np.int64)
    assert [locate_global_window(cumulative, index) for index in range(6)] == [
        (0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 0),
    ]
    with pytest.raises(IndexError):
        locate_global_window(cumulative, 6)
