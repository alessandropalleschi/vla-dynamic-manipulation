"""Image helpers: letterboxing a native render (any aspect ratio) into the square model input."""

from __future__ import annotations

import numpy as np
from PIL import Image


def letterbox_geometry(h: int, w: int, size: int):
    """(new_h, new_w, top, left) of an h x w image scaled to fit a size x size square, centred."""
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    return nh, nw, (size - nh) // 2, (size - nw) // 2


def letterbox(img: np.ndarray, size: int) -> np.ndarray:
    """Resize preserving the aspect ratio and pad with black to ``size`` x ``size``."""
    h, w = img.shape[:2]
    if (h, w) == (size, size):
        return img
    nh, nw, top, left = letterbox_geometry(h, w, size)
    resample = Image.LANCZOS if max(h, w) > size else Image.BILINEAR
    out = np.zeros((size, size, 3), dtype=np.uint8)
    out[top : top + nh, left : left + nw] = np.asarray(Image.fromarray(img).resize((nw, nh), resample))
    return out
