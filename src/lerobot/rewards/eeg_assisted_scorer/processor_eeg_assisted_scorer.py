# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

from typing import Any

from lerobot.lerobot_types import PolicyAction
from lerobot.processor import AddBatchDimensionProcessorStep, DeviceProcessorStep, PolicyProcessorPipeline
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_eeg_assisted_scorer import EEGAssistedScorerConfig


def make_eeg_assisted_scorer_pre_post_processors(
    config: EEGAssistedScorerConfig,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Move pairwise scorer tensors to the configured device without normalization."""

    del dataset_stats
    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=[
            AddBatchDimensionProcessorStep(),
            DeviceProcessorStep(device=config.device or "cpu", float_dtype="float32"),
        ],
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    postprocessor = PolicyProcessorPipeline(name=POLICY_POSTPROCESSOR_DEFAULT_NAME)
    return preprocessor, postprocessor
