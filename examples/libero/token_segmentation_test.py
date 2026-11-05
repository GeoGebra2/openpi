"""Regression tests for the image-token patch target mask."""

import numpy as np
import pytest

from examples.libero.run_counterfactual_image_delay import _target_mask
from examples.libero.run_token_drift_video import Frame
from examples.libero.run_token_patch_success import _token_mask


class _FakeEnv:
    instance_to_id = {"target": 1}


def test_target_mask_excludes_background_id_zero():
    raw = np.zeros((32, 32), dtype=np.int32)
    raw[10:18, 12:20] = 1
    raw[22:28, 22:28] = 2
    mask = _target_mask(
        _FakeEnv(), {"agentview_segmentation_instance": raw[..., None]},
        "agentview_image", "target", (32, 32),
    )
    assert mask.sum() == 64
    assert not mask[0, 0]
    assert not mask[6, 6]
    assert mask[14, 15]
    frame = Frame(np.zeros((32, 32, 3), dtype=np.uint8), mask, np.zeros((32, 32), dtype=bool))
    assert _token_mask(frame, "target", 4, 0.2).sum() <= 4


def test_target_mask_rejects_multichannel_segmentation():
    raw = np.zeros((32, 32, 2), dtype=np.int32)
    with pytest.raises(ValueError, match="single instance-ID plane"):
        _target_mask(
            _FakeEnv(), {"agentview_segmentation_instance": raw},
            "agentview_image", "target", (32, 32),
        )
