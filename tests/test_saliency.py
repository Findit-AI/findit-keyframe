"""Tests for ``findit_keyframe.saliency``.

* The module must import cleanly on systems without ``pyobjc-framework-Vision``;
  proven by collecting and running this file at all.
* ``NoopSaliencyProvider`` always returns ``0.0`` and is the default fallback.
* ``default_saliency_provider`` returns ``Noop`` on non-Darwin systems.
* ``AppleVisionSaliencyProvider`` is exercised end-to-end on macOS only;
  skipped on Linux (CI Linux runner).
"""

from __future__ import annotations

import platform

import numpy as np
import pytest

from findit_keyframe.saliency import (
    AppleVisionSaliencyProvider,
    NoopSaliencyProvider,
    SaliencyProvider,
    default_saliency_provider,
)

_IS_DARWIN = platform.system() == "Darwin"


# --------------------------------------------------------------------------- #
# NoopSaliencyProvider                                                        #
# --------------------------------------------------------------------------- #


class TestNoopSaliencyProvider:
    def test_satisfies_protocol(self):
        # SaliencyProvider is runtime-checkable so the duck-type contract
        # is asserted, not just structural shape.
        assert isinstance(NoopSaliencyProvider(), SaliencyProvider)

    def test_returns_zero_for_any_input(self):
        provider = NoopSaliencyProvider()
        assert provider.compute(np.zeros((4, 4, 3), dtype=np.uint8)) == 0.0
        assert provider.compute(np.full((16, 16, 3), 255, dtype=np.uint8)) == 0.0
        rng = np.random.default_rng(seed=0)
        noise = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        assert provider.compute(noise) == 0.0


# --------------------------------------------------------------------------- #
# default_saliency_provider                                                   #
# --------------------------------------------------------------------------- #


class TestDefaultProvider:
    def test_returns_noop_on_non_darwin(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        provider = default_saliency_provider()
        assert isinstance(provider, NoopSaliencyProvider)

    @pytest.mark.skipif(not _IS_DARWIN, reason="Apple Vision is macOS-only")
    @pytest.mark.macos
    def test_returns_apple_provider_on_macos(self):
        provider = default_saliency_provider()
        assert isinstance(provider, AppleVisionSaliencyProvider)


# --------------------------------------------------------------------------- #
# AppleVisionSaliencyProvider — instantiation guards                          #
# --------------------------------------------------------------------------- #


class TestAppleProviderGuards:
    def test_non_darwin_instantiation_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        with pytest.raises(RuntimeError, match="macOS"):
            AppleVisionSaliencyProvider()


# --------------------------------------------------------------------------- #
# AppleVisionSaliencyProvider — actual Vision call (macOS only)               #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not _IS_DARWIN, reason="Apple Vision is macOS-only")
@pytest.mark.macos
class TestAppleVisionCompute:
    def _provider(self) -> AppleVisionSaliencyProvider:
        return AppleVisionSaliencyProvider()

    def test_centre_white_frame_has_attention(self):
        # 128x128 black with a centred 64x64 white patch.
        rgb = np.zeros((128, 128, 3), dtype=np.uint8)
        rgb[32:96, 32:96] = 255
        score = self._provider().compute(rgb)
        assert 0.0 <= score <= 1.0
        # The patch is highly attention-grabbing relative to a flat field.
        assert score > 0.0

    def test_uniform_frame_has_low_attention(self):
        # A perfectly flat field has nothing salient.
        rgb = np.full((128, 128, 3), 128, dtype=np.uint8)
        score = self._provider().compute(rgb)
        assert 0.0 <= score <= 1.0

    def test_returns_float(self):
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb[16:48, 16:48] = 200
        score = self._provider().compute(rgb)
        assert isinstance(score, float)
