"""Optional saliency providers.

The sampler can score candidates with a per-frame "saliency mass" in
``[0.0, 1.0]`` reflecting how attention-grabbing the frame is. Two providers
ship today:

* ``NoopSaliencyProvider`` — always returns ``0.0``. Default and fallback.
* ``AppleVisionSaliencyProvider`` — wraps Apple's ``VNGenerateAttention\
BasedSaliencyImageRequest`` (macOS only, requires ``pyobjc-framework-Vision``
from the ``[macos]`` extra).

The Apple provider derives its scalar from the request's ``salientObjects``
bounding boxes — ``sum(area * confidence)`` clamped to ``[0, 1]`` — rather
than reading the raw saliency heatmap ``CVPixelBuffer``. The heatmap path
needs awkward ctypes pointer dereferences from pyobjc; bounding boxes carry
the same "is anything attention-grabbing here" signal with a clean API. The
Rust port (``objc2-vision``) can use either path.
"""

from __future__ import annotations

import platform
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

__all__ = [
    "AppleVisionSaliencyProvider",
    "NoopSaliencyProvider",
    "SaliencyProvider",
    "default_saliency_provider",
]


@runtime_checkable
class SaliencyProvider(Protocol):
    """Compute a per-frame saliency mass in ``[0.0, 1.0]`` for an RGB array."""

    def compute(self, rgb: npt.NDArray[np.uint8]) -> float: ...


class NoopSaliencyProvider:
    """Always returns ``0.0``. Used as the default and on non-macOS systems."""

    def compute(self, rgb: npt.NDArray[np.uint8]) -> float:
        return 0.0


class AppleVisionSaliencyProvider:
    """Apple Vision attention-based saliency. macOS only.

    ``pyobjc-framework-Vision`` and ``pyobjc-framework-Quartz`` are imported
    lazily inside ``__init__`` so the surrounding module can be imported on
    Linux (CI runners) without exploding.
    """

    def __init__(self) -> None:
        """Construct an Apple Vision saliency provider.

        Raises:
            RuntimeError: If the host is not macOS, or if
                ``pyobjc-framework-Vision`` and ``pyobjc-framework-Quartz``
                are not importable (install via the ``[macos]`` extra).
        """
        if platform.system() != "Darwin":
            raise RuntimeError("AppleVisionSaliencyProvider requires macOS")
        try:
            from Quartz import (
                CGColorSpaceCreateDeviceRGB,
                CGDataProviderCreateWithData,
                CGImageCreate,
                kCGImageAlphaNoneSkipLast,
                kCGRenderingIntentDefault,
            )
            from Vision import (
                VNGenerateAttentionBasedSaliencyImageRequest,
                VNImageRequestHandler,
            )
        except ImportError as exc:
            raise RuntimeError(
                "AppleVisionSaliencyProvider requires pyobjc-framework-Vision; "
                "install with: pip install -e '.[macos]'"
            ) from exc

        # Hold strong references on self so each compute() call avoids module
        # attribute lookups in the hot path.
        self._CGImageCreate = CGImageCreate
        self._CGDataProviderCreateWithData = CGDataProviderCreateWithData
        self._CGColorSpaceCreateDeviceRGB = CGColorSpaceCreateDeviceRGB
        self._kCGImageAlphaNoneSkipLast = kCGImageAlphaNoneSkipLast
        self._kCGRenderingIntentDefault = kCGRenderingIntentDefault
        self._VNRequest = VNGenerateAttentionBasedSaliencyImageRequest
        self._VNHandler = VNImageRequestHandler

    def _rgb_to_cgimage(self, rgb: npt.NDArray[np.uint8]) -> Any:
        """Wrap a packed RGB24 ndarray as a Core Graphics ``CGImage``.

        Quartz requires 32-bit aligned bitmap formats, so the buffer is
        padded RGB → RGBX with the alpha channel marked as ignored
        (``kCGImageAlphaNoneSkipLast``).

        Args:
            rgb: Array of shape ``(H, W, 3)`` and dtype ``uint8``.

        Returns:
            An opaque ``CGImage`` reference suitable for
            ``VNImageRequestHandler.initWithCGImage_options_``.

        Raises:
            ValueError: If ``rgb`` is not ``uint8`` or its shape is not
                ``(H, W, 3)``.
        """
        import numpy as np

        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(
                f"rgb must be uint8 (H, W, 3); got dtype={rgb.dtype}, shape={rgb.shape}"
            )
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        # Quartz wants 32-bit aligned bitmaps; pad RGB -> RGBX with alpha ignored.
        rgbx = np.empty((height, width, 4), dtype=np.uint8)
        rgbx[..., :3] = rgb
        rgbx[..., 3] = 0
        data = bytes(rgbx)
        provider = self._CGDataProviderCreateWithData(None, data, len(data), None)
        color_space = self._CGColorSpaceCreateDeviceRGB()
        return self._CGImageCreate(
            width,
            height,
            8,  # bitsPerComponent
            32,  # bitsPerPixel
            width * 4,  # bytesPerRow
            color_space,
            self._kCGImageAlphaNoneSkipLast,
            provider,
            None,  # decode
            False,  # shouldInterpolate (positional per Objective-C signature)  # noqa: FBT003
            self._kCGRenderingIntentDefault,
        )

    def compute(self, rgb: npt.NDArray[np.uint8]) -> float:
        """Run an Apple Vision attention saliency request and return its scalar.

        Each ``VNRectangleObservation`` returned by the request carries a
        normalised ``boundingBox`` (in ``[0, 1]`` image coordinates) and a
        ``confidence`` in ``[0, 1]``. The provider sums ``area *
        confidence`` over all such observations and clamps to ``[0, 1]``.
        Vision request failure (rare) collapses to ``0.0`` rather than
        raising, so a single bad frame doesn't abort a whole shot.

        Args:
            rgb: Array of shape ``(H, W, 3)`` and dtype ``uint8``.

        Returns:
            Saliency mass in ``[0.0, 1.0]``. Higher means more
            attention-grabbing.

        Raises:
            ValueError: If ``rgb`` has the wrong dtype or shape (propagated
                from :meth:`_rgb_to_cgimage`).
        """
        cg_image = self._rgb_to_cgimage(rgb)
        request = self._VNRequest.alloc().init()
        handler = self._VNHandler.alloc().initWithCGImage_options_(cg_image, {})
        success, _error = handler.performRequests_error_([request], None)
        if not success:
            return 0.0
        results = request.results() or []
        if not results:
            return 0.0
        observation = results[0]
        salient_objects = observation.salientObjects() or []
        total = 0.0
        for obj in salient_objects:
            box = obj.boundingBox()
            area = float(box.size.width) * float(box.size.height)
            confidence = float(obj.confidence())
            total += area * confidence
        return min(1.0, total)


def default_saliency_provider() -> SaliencyProvider:
    """Return the best provider available on the current platform.

    Returns:
        :class:`AppleVisionSaliencyProvider` on macOS when
        ``pyobjc-framework-Vision`` is importable; :class:`NoopSaliencyProvider`
        otherwise (including macOS hosts without the ``[macos]`` extra).
    """
    if platform.system() == "Darwin":
        try:
            return AppleVisionSaliencyProvider()
        except RuntimeError:
            return NoopSaliencyProvider()
    return NoopSaliencyProvider()
