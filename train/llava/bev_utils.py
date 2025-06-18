"""BEV token utilities

This file defines helper functions for encoding/decoding bird-eye-view (BEV)
coordinates to special tokens that can be learned by a language model.  The
new *shared-token* design uses *one* set of tokens for both **x** and **y**
axes (instead of the previous Cartesian product), which drastically reduces
the vocabulary size.

Resolution
----------
The grid resolution can be configured through the environment variable
``BEV_RESOLUTION`` (default **0.3 m**).  Practical choices are 0.3 m or 0.6 m
as requested.
"""

import numpy as np
import re
import os

# -----------------------------------------------------------------------------
# 1.  Discretisation grid
# -----------------------------------------------------------------------------

# Spatial extent (metres)
_X_MIN, _X_MAX = -100.0, 100.0   # left/right (forward is +x)
_Y_MIN, _Y_MAX = -100.0, 100.0   # lateral axis

# Resolution (metres).  Can be set at runtime via the BEV_RESOLUTION env-var.
_RES = float(os.getenv("BEV_RESOLUTION", "0.3"))  # default 0.3 m

# Generate equally-spaced bin centres for x and y.
x_bins = np.arange(_X_MIN, _X_MAX, _RES)
y_bins = np.arange(_Y_MIN, _Y_MAX, _RES)

# -----------------------------------------------------------------------------
# 2.  Token helpers
# -----------------------------------------------------------------------------

def bev_tokens():
    """Return the list of **shared** BEV special tokens.

    The same token set is re-used for both *x* and *y* coordinates.  Only the
    maximum of ``len(x_bins)`` and ``len(y_bins)`` tokens is required.
    """
    n_tokens = max(len(x_bins), len(y_bins))
    return [f"<bev_{i}>" for i in range(n_tokens)]

def clamp_index(value, bins):
    """Clip value to range and return nearest index."""
    idx = np.argmin(np.abs(bins - value))
    return int(idx)

def coord_to_token(x: float, y: float) -> str:
    """Convert a real-world (x, y) pair to a pair of BEV tokens.

    The returned **string** contains *two* tokens separated by a space,
    suitable for direct insertion into natural-language prompts.
    """
    xi = clamp_index(x, x_bins)
    yi = clamp_index(y, y_bins)
    return f"<bev_{xi}> <bev_{yi}>"

# ---------------------------------------------------------------------
# Decoding helpers – transform BEV tokens back to real‑world coordinates
# ---------------------------------------------------------------------

_BEV_REGEX = re.compile(r"<bev_(\d+)>")

# The old *token_to_coord* API is now deprecated because one token no longer
# contains the full (x, y) pair.  It is kept for backward compatibility **only
# to decode individual axis tokens**, raising if the axis cannot be inferred.

def token_to_value(token: str, axis: str = "x") -> float:
    """Return the *x* or *y* coordinate value represented by *token*."""
    m = _BEV_REGEX.fullmatch(token)
    if m is None:
        raise ValueError(f"Invalid BEV token: {token}")
    idx = int(m.group(1))
    bins = x_bins if axis.lower() == "x" else y_bins
    if idx >= len(bins):
        raise IndexError(f"BEV index out of range in token {token}")
    return float(bins[idx])

def decode_bev_tokens(text: str) -> list[tuple[float, float]]:
    """Extract and decode BEV tokens from *text*.

    Tokens are expected **in order**: x-token, y-token, x-token, y-token, …
    (possibly mixed with normal words).  The function pairs them sequentially
    and returns a list of (x, y) tuples.  If the total number of tokens is
    odd, the last stray token is ignored.
    """
    indices = [int(m.group(1)) for m in _BEV_REGEX.finditer(text)]
    coords: list[tuple[float, float]] = []
    for i in range(0, len(indices) - 1, 2):  # step of 2, ignore last if odd
        xi, yi = indices[i], indices[i + 1]
        if xi >= len(x_bins) or yi >= len(y_bins):
            raise IndexError(f"BEV index out of range in tokens <bev_{xi}>, <bev_{yi}>")
        coords.append((float(x_bins[xi]), float(y_bins[yi])))
    return coords
