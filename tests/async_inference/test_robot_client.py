# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit-tests for the `RobotClient` action-queue logic (pure Python, no gRPC).

We monkey-patch `lerobot.robots.utils.make_robot_from_config` so that
no real hardware is accessed. Only the queue-update mechanism is verified.
"""

from __future__ import annotations

import threading
import time
from queue import Queue

import pytest
import torch

# Skip entire module if required deps are not available
pytest.importorskip("grpc")
pytest.importorskip("serial", reason="pyserial is required (install lerobot[hardware])")
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------


@pytest.fixture()
def robot_client():
    """Fresh `RobotClient` instance for each test case (no threads started).
    Uses DummyRobot."""
    # Import only when the test actually runs (after decorator check)
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.robot_client import RobotClient
    from tests.mocks.mock_robot import MockRobotConfig

    test_config = MockRobotConfig()

    # gRPC channel is not actually used in tests, so using a dummy address
    test_config = RobotClientConfig(
        robot=test_config,
        server_address="localhost:9999",
        policy_type="test",
        pretrained_name_or_path="test",
        actions_per_chunk=20,
    )

    client = RobotClient(test_config)

    # Initialize attributes that are normally set in start() method
    client.chunks_received = 0
    client.available_actions_size = []

    yield client

    if client.robot.is_connected:
        client.stop()


# -----------------------------------------------------------------------------
# Helper utilities for tests
# -----------------------------------------------------------------------------


def _make_actions(start_ts: float, start_t: int, count: int):
    """Generate `count` consecutive TimedAction objects starting at timestep `start_t`."""
    from lerobot.async_inference.helpers import TimedAction

    fps = 30  # emulates most common frame-rate
    actions = []
    for i in range(count):
        timestep = start_t + i
        timestamp = start_ts + i * (1 / fps)
        action_tensor = torch.full((6,), timestep, dtype=torch.float32)
        actions.append(TimedAction(action=action_tensor, timestep=timestep, timestamp=timestamp))
    return actions


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_update_action_queue_discards_stale(robot_client):
    """`_update_action_queue` must drop actions with `timestep` <= `latest_action`."""

    # Pretend we already executed up to action #4
    robot_client.latest_action = 4

    # Incoming chunk contains timesteps 3..7 -> expect 5,6,7 kept.
    incoming = _make_actions(start_ts=time.time(), start_t=3, count=5)  # 3,4,5,6,7

    robot_client._aggregate_action_queues(incoming)

    # Extract timesteps from queue
    resulting_timesteps = [a.get_timestep() for a in robot_client.action_queue.queue]

    assert resulting_timesteps == [5, 6, 7]


@pytest.mark.parametrize(
    "weight_old, weight_new",
    [
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 0.5),
        (0.2, 0.8),
        (0.8, 0.2),
        (0.1, 0.9),
        (0.9, 0.1),
    ],
)
def test_aggregate_action_queues_combines_actions_in_overlap(
    robot_client, weight_old: float, weight_new: float
):
    """`_aggregate_action_queues` must combine actions on overlapping timesteps according
    to the provided aggregate_fn, here tested with multiple coefficients."""
    from lerobot.async_inference.helpers import TimedAction

    robot_client.chunks_received = 0

    # Pretend we already executed up to action #4, and queue contains actions for timesteps 5..6
    robot_client.latest_action = 4
    current_actions = _make_actions(
        start_ts=time.time(), start_t=5, count=2
    )  # actions are [torch.ones(6), torch.ones(6), ...]
    current_actions = [
        TimedAction(action=10 * a.get_action(), timestep=a.get_timestep(), timestamp=a.get_timestamp())
        for a in current_actions
    ]

    for a in current_actions:
        robot_client.action_queue.put(a)

    # Incoming chunk contains timesteps 3..7 -> expect 5,6,7 kept.
    incoming = _make_actions(start_ts=time.time(), start_t=3, count=5)  # 3,4,5,6,7

    overlap_timesteps = [5, 6]  # properly tested in test_aggregate_action_queues_discards_stale
    nonoverlap_timesteps = [7]

    robot_client._aggregate_action_queues(
        incoming, aggregate_fn=lambda x1, x2: weight_old * x1 + weight_new * x2
    )

    queue_overlap_actions = []
    queue_non_overlap_actions = []
    for a in robot_client.action_queue.queue:
        if a.get_timestep() in overlap_timesteps:
            queue_overlap_actions.append(a)
        elif a.get_timestep() in nonoverlap_timesteps:
            queue_non_overlap_actions.append(a)

    queue_overlap_actions = sorted(queue_overlap_actions, key=lambda x: x.get_timestep())
    queue_non_overlap_actions = sorted(queue_non_overlap_actions, key=lambda x: x.get_timestep())

    assert torch.allclose(
        queue_overlap_actions[0].get_action(),
        weight_old * current_actions[0].get_action() + weight_new * incoming[-3].get_action(),
    )
    assert torch.allclose(
        queue_overlap_actions[1].get_action(),
        weight_old * current_actions[1].get_action() + weight_new * incoming[-2].get_action(),
    )
    assert torch.allclose(queue_non_overlap_actions[0].get_action(), incoming[-1].get_action())


@pytest.mark.parametrize(
    "chunk_size, queue_len, expected",
    [
        (20, 12, False),  # 12 / 20 = 0.6  > g=0.5 threshold, not ready to send
        (20, 8, True),  # 8  / 20 = 0.4 <= g=0.5, ready to send
        (10, 5, True),
        (10, 6, False),
    ],
)
def test_ready_to_send_observation(robot_client, chunk_size: int, queue_len: int, expected: bool):
    """Validate `_ready_to_send_observation` ratio logic for various sizes."""

    robot_client.action_chunk_size = chunk_size

    # Clear any existing actions then fill with `queue_len` dummy entries ----
    robot_client.action_queue = Queue()

    dummy_actions = _make_actions(start_ts=time.time(), start_t=0, count=queue_len)
    for act in dummy_actions:
        robot_client.action_queue.put(act)

    assert robot_client._ready_to_send_observation() is expected


@pytest.mark.parametrize(
    "g_threshold, expected",
    [
        # The condition is `queue_size / chunk_size <= g`.
        # Here, ratio = 6 / 10 = 0.6.
        (0.0, False),  # 0.6 <= 0.0 is False
        (0.1, False),
        (0.2, False),
        (0.3, False),
        (0.4, False),
        (0.5, False),
        (0.6, True),  # 0.6 <= 0.6 is True
        (0.7, True),
        (0.8, True),
        (0.9, True),
        (1.0, True),
    ],
)
def test_ready_to_send_observation_with_varying_threshold(robot_client, g_threshold: float, expected: bool):
    """Validate `_ready_to_send_observation` with fixed sizes and varying `g`."""
    # Fixed sizes for this test: ratio = 6 / 10 = 0.6
    chunk_size = 10
    queue_len = 6

    robot_client.action_chunk_size = chunk_size
    # This is the parameter we are testing
    robot_client._chunk_size_threshold = g_threshold

    # Fill queue with dummy actions
    robot_client.action_queue = Queue()
    dummy_actions = _make_actions(start_ts=time.time(), start_t=0, count=queue_len)
    for act in dummy_actions:
        robot_client.action_queue.put(act)

    assert robot_client._ready_to_send_observation() is expected


def test_ready_to_send_observation_allows_only_one_inference_request(robot_client):
    robot_client.action_chunk_size = 20
    robot_client.action_queue = Queue()
    for action in _make_actions(start_ts=time.time(), start_t=0, count=8):
        robot_client.action_queue.put(action)

    with robot_client._inference_request_lock:
        robot_client._inference_request_started_at = time.monotonic()

    assert robot_client._ready_to_send_observation() is False


def test_expired_inference_request_can_be_retried(robot_client):
    robot_client.action_chunk_size = 20
    robot_client.action_queue = Queue()
    for action in _make_actions(start_ts=time.time(), start_t=0, count=8):
        robot_client.action_queue.put(action)
    with robot_client._inference_request_lock:
        robot_client._inference_request_started_at = (
            time.monotonic() - robot_client.config.inference_request_timeout_s - 0.1
        )

    assert robot_client._ready_to_send_observation() is True
    assert robot_client._inference_request_started_at is None


def test_chunk_transition_preserves_prefix_then_smoothly_blends(robot_client):
    from lerobot.async_inference.helpers import TimedAction

    robot_client.config.action_commit_steps = 2
    robot_client.config.action_blend_steps = 3
    robot_client.latest_action = 4
    now = time.time()
    for timestep in range(5, 10):
        robot_client.action_queue.put(TimedAction(timestamp=now, timestep=timestep, action=torch.zeros(6)))
    incoming = [
        TimedAction(timestamp=now, timestep=timestep, action=torch.full((6,), 10.0))
        for timestep in range(5, 11)
    ]

    robot_client._aggregate_action_queues(incoming, aggregate_fn=lambda _old, new: new)

    actions = {action.get_timestep(): action.get_action() for action in robot_client.action_queue.queue}
    assert torch.allclose(actions[5], torch.zeros(6))
    assert torch.allclose(actions[6], torch.zeros(6))
    assert torch.allclose(actions[7], torch.full((6,), 10.0 * (7 / 27)))
    assert torch.allclose(actions[8], torch.full((6,), 10.0 * (20 / 27)))
    assert torch.allclose(actions[9], torch.full((6,), 10.0))
    assert torch.allclose(actions[10], torch.full((6,), 10.0))


def test_action_tensor_dimension_must_match_robot_joint_interface(robot_client):
    expected_dim = len(robot_client.robot.action_features)

    with pytest.raises(ValueError, match=f"flat {expected_dim}D tensor"):
        robot_client._action_tensor_to_action_dict(torch.zeros(expected_dim + 1))


def test_control_loop_can_bound_number_of_policy_actions(robot_client, monkeypatch):
    performed = []
    robot_client.start_barrier = type("BarrierStub", (), {"wait": lambda self: None})()
    monkeypatch.setattr(robot_client, "actions_available", lambda: True)
    monkeypatch.setattr(
        robot_client,
        "control_loop_action",
        lambda verbose=False: performed.append(len(performed)) or {"action": len(performed)},
    )
    monkeypatch.setattr(robot_client, "_ready_to_send_observation", lambda: False)
    monkeypatch.setattr("lerobot.async_inference.robot_client.time.sleep", lambda _: None)

    robot_client.control_loop(task="test", max_actions=3)

    assert len(performed) == 3


def test_control_loop_records_pre_action_observation_and_performed_action(robot_client, monkeypatch):
    recorded = []
    observation = {"state": 1.0}
    timestamps = {"state": 123}
    robot_client.start_barrier = type("BarrierStub", (), {"wait": lambda self: None})()
    monkeypatch.setattr(robot_client, "actions_available", lambda: True)
    monkeypatch.setattr(robot_client.robot, "get_observation", lambda: observation)
    monkeypatch.setattr(robot_client.robot, "last_observation_timestamps_ns", timestamps, raising=False)
    monkeypatch.setattr(
        robot_client,
        "control_loop_action",
        lambda verbose=False: {"safe_action": 2.0},
    )
    monkeypatch.setattr(robot_client, "_ready_to_send_observation", lambda: False)
    monkeypatch.setattr("lerobot.async_inference.robot_client.time.sleep", lambda _: None)

    robot_client.control_loop(
        task="test",
        max_actions=1,
        frame_callback=lambda obs, action, stamps, action_time: recorded.append(
            (obs, action, stamps, action_time)
        ),
    )

    assert recorded[0][:3] == (observation, {"safe_action": 2.0}, timestamps)
    assert isinstance(recorded[0][3], int)


def test_control_loop_stop_event_prevents_next_action(robot_client, monkeypatch):
    stop_event = threading.Event()
    stop_event.set()
    robot_client.start_barrier = type("BarrierStub", (), {"wait": lambda self: None})()
    monkeypatch.setattr(
        robot_client,
        "control_loop_action",
        lambda verbose=False: pytest.fail("an action was executed after stop"),
    )

    robot_client.control_loop(task="test", stop_event=stop_event)


def test_control_loop_rejects_nonpositive_action_bound(robot_client):
    with pytest.raises(ValueError, match="max_actions must be positive"):
        robot_client.control_loop(task="test", max_actions=0)


# -----------------------------------------------------------------------------
# Regression test: robot type registry populated by robot_client imports
# -----------------------------------------------------------------------------


def test_robot_client_registers_builtin_robot_types():
    """Importing robot_client must populate RobotConfig's ChoiceRegistry.

    This is a regression test for a bug introduced in #2425, where removing
    robot module imports from robot_client.py caused RobotConfig's registry to
    be empty, breaking CLI argument parsing with:
      error: argument --robot.type: invalid choice: 'so101_follower' (choose from )

    Robot types are registered via @RobotConfig.register_subclass() decorators
    at import time, so all supported modules must be explicitly imported.
    """
    import lerobot.async_inference.robot_client  # noqa: F401
    from lerobot.robots.config import RobotConfig

    known_choices = RobotConfig.get_known_choices()

    expected_robot_types = [
        "so100_follower",
        "so101_follower",
        "koch_follower",
        "omx_follower",
        "bi_so_follower",
    ]
    for robot_type in expected_robot_types:
        assert robot_type in known_choices, (
            f"Robot type '{robot_type}' is not registered in RobotConfig's ChoiceRegistry. "
            f"Ensure the corresponding module is imported in robot_client.py. "
            f"Known choices: {sorted(known_choices)}"
        )
