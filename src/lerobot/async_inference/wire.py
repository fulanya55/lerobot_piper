# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Stable async-inference wire payloads shared across LeRobot forks.

The transport remains pickle-based for tensor support, but only built-in containers and
tensors are placed on the wire. This avoids coupling clients to the Python module path of
``RemotePolicyConfig``, ``TimedObservation``, and ``TimedAction``.
"""

from __future__ import annotations

import pickle  # nosec B403: internal trusted-network transport, matching the existing protocol
from typing import Any

from lerobot.configs import FeatureType, PolicyFeature

from .helpers import RemotePolicyConfig, TimedAction, TimedObservation

WIRE_MARKER = "__lerobot_async_inference_wire__"
WIRE_VERSION = 1


def _envelope(kind: str, payload: Any) -> dict[str, Any]:
    return {WIRE_MARKER: WIRE_VERSION, "kind": kind, "payload": payload}


def _unpack(data: bytes, expected_kind: str) -> Any:
    value = pickle.loads(data)  # nosec B301: existing protocol assumes a trusted peer
    if not isinstance(value, dict) or value.get(WIRE_MARKER) != WIRE_VERSION:
        return value
    if value.get("kind") != expected_kind:
        raise ValueError(f"Expected wire payload {expected_kind!r}, got {value.get('kind')!r}")
    return value["payload"]


def serialize_policy_config(config: RemotePolicyConfig) -> bytes:
    features = {}
    for key, feature in config.lerobot_features.items():
        if isinstance(feature, PolicyFeature):
            features[key] = {
                "__policy_feature__": True,
                "type": feature.type.value,
                "shape": tuple(feature.shape),
            }
        else:
            features[key] = feature
    payload = {
        "policy_type": config.policy_type,
        "pretrained_name_or_path": config.pretrained_name_or_path,
        "lerobot_features": features,
        "actions_per_chunk": config.actions_per_chunk,
        "device": config.device,
        "rename_map": config.rename_map,
    }
    return pickle.dumps(_envelope("policy_setup", payload))  # nosec B301


def deserialize_policy_config(data: bytes) -> RemotePolicyConfig:
    payload = _unpack(data, "policy_setup")
    if isinstance(payload, RemotePolicyConfig):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"Policy setup must be a mapping, got {type(payload)}")
    features = {
        key: (
            PolicyFeature(type=FeatureType(feature["type"]), shape=tuple(feature["shape"]))
            if feature.get("__policy_feature__")
            else feature
        )
        for key, feature in payload["lerobot_features"].items()
    }
    return RemotePolicyConfig(
        policy_type=payload["policy_type"],
        pretrained_name_or_path=payload["pretrained_name_or_path"],
        lerobot_features=features,
        actions_per_chunk=int(payload["actions_per_chunk"]),
        device=payload.get("device", "cpu"),
        rename_map=payload.get("rename_map", {}),
    )


def serialize_timed_observation(observation: TimedObservation) -> bytes:
    payload = {
        "timestamp": observation.timestamp,
        "timestep": observation.timestep,
        "observation": observation.observation,
        "must_go": observation.must_go,
    }
    return pickle.dumps(_envelope("observation", payload))  # nosec B301


def deserialize_timed_observation(data: bytes) -> TimedObservation:
    payload = _unpack(data, "observation")
    if isinstance(payload, TimedObservation):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"Observation must be a mapping, got {type(payload)}")
    return TimedObservation(
        timestamp=float(payload["timestamp"]),
        timestep=int(payload["timestep"]),
        observation=payload["observation"],
        must_go=bool(payload.get("must_go", False)),
    )


def serialize_timed_actions(actions: list[TimedAction]) -> bytes:
    payload = [
        {"timestamp": action.timestamp, "timestep": action.timestep, "action": action.action}
        for action in actions
    ]
    return pickle.dumps(_envelope("actions", payload))  # nosec B301


def deserialize_timed_actions(data: bytes) -> list[TimedAction]:
    payload = _unpack(data, "actions")
    if isinstance(payload, list) and all(isinstance(action, TimedAction) for action in payload):
        return payload
    if not isinstance(payload, list) or not all(isinstance(action, dict) for action in payload):
        raise TypeError("Actions must be a list of mappings")
    return [
        TimedAction(
            timestamp=float(action["timestamp"]),
            timestep=int(action["timestep"]),
            action=action["action"],
        )
        for action in payload
    ]
