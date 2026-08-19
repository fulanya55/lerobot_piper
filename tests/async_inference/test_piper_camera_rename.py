import numpy as np

from lerobot.async_inference.helpers import prepare_raw_observation
from lerobot.configs import FeatureType, PolicyFeature


def test_prepare_raw_observation_resizes_using_checkpoint_rename_map():
    robot_features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["joint_1", "joint_2"],
        },
        "observation.images.cam_high_rgb": {
            "dtype": "image",
            "shape": (4, 6, 3),
            "names": ["height", "width", "channels"],
        },
    }
    robot_observation = {
        "joint_1": 0.1,
        "joint_2": 0.2,
        "cam_high_rgb": np.zeros((4, 6, 3), dtype=np.uint8),
    }
    policy_images = {"observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 2, 3))}

    prepared = prepare_raw_observation(
        robot_observation,
        robot_features,
        policy_images,
        observation_rename_map={"observation.images.cam_high_rgb": "observation.images.base_0_rgb"},
    )

    assert prepared["observation.images.cam_high_rgb"].shape == (3, 2, 3)
