#!/usr/bin/env python

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

import re

from lerobot.common.wandb_utils import get_safe_wandb_artifact_name


def test_get_safe_wandb_artifact_name_sanitizes_multi_dataset_name():
    name = "policy_pi0-seed_42-dataset_['dataset/a', 'dataset b']-005000"

    safe_name = get_safe_wandb_artifact_name(name)

    assert re.fullmatch(r"[a-zA-Z0-9_.-]+", safe_name)
    assert safe_name.endswith("-005000")


def test_get_safe_wandb_artifact_name_truncates_deterministically():
    name = f"policy_pi0-dataset_{['dataset'] * 100}-005000"

    safe_name = get_safe_wandb_artifact_name(name)

    assert len(safe_name) == 128
    assert safe_name == get_safe_wandb_artifact_name(name)
    assert safe_name.endswith("-005000")
