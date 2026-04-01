"""Tests for Hydra YAML export."""

from __future__ import annotations

import torchio as tio


class TestToHydra:
    def test_noise_default(self) -> None:
        n = tio.Noise()
        cfg = n.to_hydra()
        assert cfg["_target_"] == "torchio.Noise"
        assert "std" not in cfg  # default, omitted

    def test_noise_custom(self) -> None:
        n = tio.Noise(std=(0.05, 0.2), p=0.5)
        cfg = n.to_hydra()
        assert cfg["_target_"] == "torchio.Noise"
        assert cfg["std"] == [0.05, 0.2]
        assert cfg["p"] == 0.5
        assert "mean" not in cfg  # still default

    def test_flip(self) -> None:
        f = tio.Flip(axes=(0, 1))
        cfg = f.to_hydra()
        assert cfg["_target_"] == "torchio.Flip"
        assert cfg["axes"] == [0, 1]

    def test_compose(self) -> None:
        pipeline = tio.Compose(
            [
                tio.Flip(axes=(0,), p=0.5),
                tio.Noise(std=0.1),
            ]
        )
        cfg = pipeline.to_hydra()
        assert cfg["_target_"] == "torchio.Compose"
        assert len(cfg["transforms"]) == 2
        assert cfg["transforms"][0]["_target_"] == "torchio.Flip"
        assert cfg["transforms"][1]["_target_"] == "torchio.Noise"

    def test_nested_compose(self) -> None:
        pipeline = tio.Compose(
            [
                tio.OneOf([tio.Noise(), tio.Flip()]),
            ]
        )
        cfg = pipeline.to_hydra()
        inner = cfg["transforms"][0]
        assert inner["_target_"] == "torchio.OneOf"
        assert "transforms" in inner

    def test_round_trip_values(self) -> None:
        """Hydra config values should be plain Python types."""
        n = tio.Noise(std=(0.05, 0.2), mean=0.5, p=0.8)
        cfg = n.to_hydra()
        # All values should be JSON-compatible types
        for v in cfg.values():
            assert isinstance(v, (str, int, float, bool, list, type(None)))
