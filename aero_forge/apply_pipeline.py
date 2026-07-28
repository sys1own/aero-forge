"""RGBA pixel pipeline with defensive bounds handling.

The pipeline accepts packed RGBA buffers (or RGB buffers with missing alpha)
and applies simple per-pixel transforms. It is designed to be robust against
short buffers, zero-size images, and boundary-length mismatches.
"""

from __future__ import annotations

from typing import List, Sequence, Union

PixelBuffer = Union[Sequence[int], bytes, bytearray]


def _normalize_pixels(pixels: PixelBuffer, width: int, height: int) -> List[int]:
    """Return a list of ``width * height * 4`` RGBA integers.

    Missing alpha values are padded with ``255``; trailing bytes beyond the
    expected size are ignored. The function never raises on short buffers.
    """
    if width <= 0 or height <= 0:
        return []

    expected = width * height * 4
    if isinstance(pixels, (bytes, bytearray)):
        data = list(pixels)
    else:
        data = [int(v) & 0xFF for v in pixels]

    if len(data) < expected:
        data.extend([255] * (expected - len(data)))
    elif len(data) > expected:
        data = data[:expected]

    return data


def apply_pipeline(
    pixels: PixelBuffer,
    width: int,
    height: int,
    operation: str = "grayscale",
) -> List[int]:
    """Apply *operation* to every pixel in the packed RGBA buffer.

    Supported operations: ``grayscale``, ``invert``, ``noop``.

    Parameters
    ----------
    pixels:
        Packed RGBA values. RGB buffers are treated as opaque (alpha=255).
    width, height:
        Image dimensions. Zero or negative dimensions return an empty list.
    operation:
        Pixel transform to apply.

    Returns
    -------
    A new packed RGBA integer list of length ``width * height * 4``.
    """
    data = _normalize_pixels(pixels, width, height)
    if not data:
        return []

    result: List[int] = []
    for i in range(0, len(data), 4):
        r, g, b, a = data[i], data[i + 1], data[i + 2], data[i + 3]
        if operation == "grayscale":
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            result.extend([gray, gray, gray, a])
        elif operation == "invert":
            result.extend([255 - r, 255 - g, 255 - b, a])
        else:  # noop or unknown
            result.extend([r, g, b, a])
    return result
