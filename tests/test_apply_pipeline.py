"""Unit tests for the RGBA apply_pipeline helper."""

from __future__ import annotations

import pytest

from aero_forge.apply_pipeline import apply_pipeline


def test_rgba_buffer_full():
    """A 2x2 red buffer with alpha 128 is processed without errors."""
    pixels = [255, 0, 0, 128] * 4
    result = apply_pipeline(pixels, 2, 2, "grayscale")
    assert len(result) == 16
    # Alpha preserved; all RGB channels equal luminance.
    assert result[3] == 128
    assert result[0] == result[1] == result[2]


def test_rgb_missing_alpha_padded():
    """A 3-element RGB array for a 1x1 image is padded to opaque alpha."""
    pixels = [1, 1, 1]
    result = apply_pipeline(pixels, 1, 1, "grayscale")
    assert len(result) == 4
    assert result[3] == 255
    gray = result[0]
    assert result[1] == gray
    assert result[2] == gray


def test_zero_dimensions():
    """Zero width or height yields an empty result, not an IndexError."""
    assert apply_pipeline([1, 2, 3, 4], 0, 1) == []
    assert apply_pipeline([1, 2, 3, 4], 1, 0) == []
    assert apply_pipeline([1, 2, 3, 4], 0, 0) == []


def test_single_pixel():
    """A single RGBA pixel is transformed correctly."""
    result = apply_pipeline([10, 20, 30, 40], 1, 1, "grayscale")
    assert len(result) == 4
    assert result[0] == result[1] == result[2]
    assert result[3] == 40


def test_invert_operation():
    """Invert subtracts each channel from 255 and preserves alpha."""
    result = apply_pipeline([10, 20, 30, 40], 1, 1, "invert")
    assert result == [245, 235, 225, 40]


def test_noop_operation():
    """Noop returns a normalized copy of the input."""
    pixels = [1, 2, 3]
    result = apply_pipeline(pixels, 1, 1, "noop")
    assert result == [1, 2, 3, 255]


def test_boundary_extra_bytes_truncated():
    """Extra bytes beyond the expected pixel count are ignored."""
    pixels = [10, 20, 30, 40, 99, 99]
    result = apply_pipeline(pixels, 1, 1, "noop")
    assert len(result) == 4
    assert result[:3] == [10, 20, 30]
    assert result[3] == 40


def test_short_buffer_multiple_pixels():
    """A buffer that is short for the requested dimensions is padded."""
    pixels = [255, 0, 0]
    result = apply_pipeline(pixels, 2, 2, "grayscale")
    assert len(result) == 16
    # Every pixel should be opaque after padding, alpha = 255.
    assert result[3::4] == [255, 255, 255, 255]


def test_bytes_input():
    """Bytes and bytearray inputs are accepted."""
    result = apply_pipeline(bytes([10, 20, 30]), 1, 1, "noop")
    assert result == [10, 20, 30, 255]

    result = apply_pipeline(bytearray([10, 20, 30]), 1, 1, "noop")
    assert result == [10, 20, 30, 255]


def test_negative_dimension_is_empty():
    """Negative dimensions are treated as an empty image."""
    assert apply_pipeline([255, 0, 0, 128], -1, 1) == []
