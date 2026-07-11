from __future__ import annotations

import unittest

import torch

from utils.data import EventFrameTransform


class EventFrameTransformTest(unittest.TestCase):
    def _transform(self, mode: str, downsample_size: int | None, frames: torch.Tensor) -> EventFrameTransform:
        transform = EventFrameTransform(
            sensor_size=(4, 4, 2),
            tmax=frames.shape[0],
            frame_mode=mode,
            downsample_size=downsample_size,
            expected_channels=frames.shape[1],
        )
        transform.to_frame = lambda _: frames.numpy()
        return transform

    def test_native_resolution_is_unchanged(self) -> None:
        frames = torch.zeros(2, 2, 4, 4)
        output = self._transform("binary", None, frames)(object())
        self.assertEqual(tuple(output.shape), (2, 2, 4, 4))

    def test_binary_downsampling_uses_region_presence(self) -> None:
        frames = torch.zeros(1, 1, 4, 4)
        frames[0, 0, 0, 0] = 3.0
        output = self._transform("binary", 2, frames)(object())
        self.assertEqual(float(output.sum()), 1.0)
        self.assertTrue(torch.all((output == 0) | (output == 1)))

    def test_count_downsampling_preserves_total_count(self) -> None:
        frames = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 4, 4)
        output = self._transform("count", 2, frames)(object())
        self.assertTrue(torch.allclose(output.sum(), frames.sum()))


if __name__ == "__main__":
    unittest.main()
