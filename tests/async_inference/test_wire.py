import torch

from lerobot.async_inference.helpers import RemotePolicyConfig, TimedAction, TimedObservation
from lerobot.async_inference.wire import (
    deserialize_policy_config,
    deserialize_timed_actions,
    deserialize_timed_observation,
    serialize_policy_config,
    serialize_timed_actions,
    serialize_timed_observation,
)
from lerobot.configs import FeatureType, PolicyFeature


def test_wire_round_trip_uses_repository_independent_payloads() -> None:
    config = RemotePolicyConfig(
        policy_type="pi0",
        pretrained_name_or_path="/models/pi0",
        lerobot_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        },
        actions_per_chunk=5,
        device="cuda",
        rename_map={"observation.images.high": "observation.images.base_0_rgb"},
    )
    decoded_config = deserialize_policy_config(serialize_policy_config(config))
    assert decoded_config == config

    observation = TimedObservation(
        timestamp=1.25,
        timestep=7,
        observation={"joint": 0.5, "task": "place block"},
        must_go=True,
    )
    decoded_observation = deserialize_timed_observation(serialize_timed_observation(observation))
    assert decoded_observation == observation

    actions = [TimedAction(timestamp=1.5, timestep=8, action=torch.tensor([0.1, 0.2]))]
    decoded_actions = deserialize_timed_actions(serialize_timed_actions(actions))
    assert decoded_actions[0].timestamp == actions[0].timestamp
    assert decoded_actions[0].timestep == actions[0].timestep
    torch.testing.assert_close(decoded_actions[0].action, actions[0].action)
