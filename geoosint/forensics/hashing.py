"""Cryptographic and perceptual hashes.

Cryptographic hashes identify the exact file (chain of custody); perceptual
hashes (aHash, dHash, pHash) survive resizing/recompression and are useful
to spot the same picture across platforms.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from ..models import Hashes


def file_hashes(path: str | Path) -> tuple[str, str, str]:
    md5, sha1, sha256 = hashlib.md5(), hashlib.sha1(), hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            md5.update(chunk)
            sha1.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha1.hexdigest(), sha256.hexdigest()


def _bits_to_hex(bits: np.ndarray) -> str:
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return f"{value:0{bits.size // 4}x}"


def _gray(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(img.convert("L").resize(size, Image.Resampling.LANCZOS), dtype=np.float64)


def average_hash(img: Image.Image, hash_size: int = 8) -> str:
    pixels = _gray(img, (hash_size, hash_size))
    return _bits_to_hex(pixels > pixels.mean())


def difference_hash(img: Image.Image, hash_size: int = 8) -> str:
    pixels = _gray(img, (hash_size + 1, hash_size))
    return _bits_to_hex(pixels[:, 1:] > pixels[:, :-1])


@lru_cache(maxsize=4)
def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)
    matrix = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    matrix[0] *= 1 / np.sqrt(2)
    return matrix * np.sqrt(2 / n)


def perceptual_hash(img: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    n = hash_size * highfreq_factor
    pixels = _gray(img, (n, n))
    dct = _dct_matrix(n)
    coeffs = dct @ pixels @ dct.T
    low = coeffs[:hash_size, :hash_size]
    return _bits_to_hex(low > np.median(low))


def hamming(hex_a: str, hex_b: str) -> int:
    """Bit distance between two hex hashes of the same length (0 = identical)."""
    return bin(int(hex_a, 16) ^ int(hex_b, 16)).count("1")


def compute_hashes(path: str | Path) -> Hashes:
    md5, sha1, sha256 = file_hashes(path)
    with Image.open(path) as img:
        img.load()
        return Hashes(
            md5=md5,
            sha1=sha1,
            sha256=sha256,
            ahash=average_hash(img),
            dhash=difference_hash(img),
            phash=perceptual_hash(img),
        )
