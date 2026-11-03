"""JPEG quantisation-table analysis."""

from __future__ import annotations

from PIL import Image

# Standard IJG (libjpeg) luminance table, quality 50.
# fmt: off
_STD_LUMA = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
]
# fmt: on


def _scaled_sum(quality: int) -> int:
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    return sum(min(255, max(1, int((v * scale + 50) // 100))) for v in _STD_LUMA)


_SUMS = {q: _scaled_sum(q) for q in range(1, 101)}


def estimate_jpeg_quality(img: Image.Image) -> int | None:
    """Estimate the IJG quality factor (1-100) of a JPEG from its luma table.

    The comparison uses the table sum, so it is independent of the
    zig-zag/natural ordering returned by Pillow.  Returns ``None`` for
    non-JPEG images.
    """
    tables = getattr(img, "quantization", None)
    if not tables:
        return None
    luma = tables.get(0)
    if not luma or len(luma) != 64:
        return None
    total = sum(luma)
    return min(_SUMS, key=lambda q: (abs(_SUMS[q] - total), -q))
