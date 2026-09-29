from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lerobot.datasets.eeg_rlhf import EEGChunkWindowDataset, EEGWindowConfig, _prepare_segment


def test_eeg_window_config_uses_two_2p5s_halves_without_cache(tmp_path: Path):
    eeg_path = tmp_path / "eeg.npz"
    sample_rate = 100.0
    timestamps = 1_000.0 + np.arange(2_000, dtype=np.float64) / sample_rate
    data = np.stack([np.sin(np.linspace(0, 30, len(timestamps))) + i for i in range(4)], axis=1).astype(np.float32)
    np.savez(
        eeg_path,
        eeg_data=data,
        eeg_channel_names=np.array(["C1", "C2", "C3", "C4"]),
        eeg_sample_rate=np.array(sample_rate),
        eeg_timestamps_pc=timestamps,
    )

    class FakeDataset:
        repo_id = "task"
        delta_timestamps = {"action": [0.0, 1.0]}

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {"timestamp": torch.tensor(5.0), "action": torch.zeros(2)}

        def get_raw_item(self, index):
            return {"timestamp": 5.0, "episode_index": 0, "frame_index": 0}

    manifest = pd.DataFrame(
        [{"lerobot_dataset": "task", "lerobot_episode_index": 0, "eeg_npz": str(eeg_path), "video_on": 997.5}]
    )
    ds = EEGChunkWindowDataset(FakeDataset(), manifest, config=EEGWindowConfig(real_channels=4, padded_channels=4))
    sample = ds[0]
    assert sample["observation.eeg_rlhf.token"].shape == (2, 4, 640)
    assert bool(sample["observation.eeg_rlhf.window_valid"])
    assert not list(tmp_path.glob("**/*cache*"))


def test_prepare_segment_pads_to_brainmu_contract():
    result = _prepare_segment(
        np.ones((2, 100), dtype=np.float32), ["C1", "C2"], 100.0, EEGWindowConfig(real_channels=2)
    )
    assert result.shape == (1, 24, 640)
