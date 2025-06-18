import numpy as np
import re

# Generate grid coordinates based on given ranges
X_GRID_RANGES = [
    (-40, -20, 2),
    (-20, 0, 1),
    (0, 20, 0.5),
    (20, 40, 1),
    (40, 80, 2),
]
Y_GRID_RANGE = (-60, 60, 1)

# Pre-compute bin centers for x and y
x_bins = []
for start, end, step in X_GRID_RANGES:
    x_bins.extend(np.arange(start, end, step).tolist())
# ensure inclusive of end?
x_bins = np.array(x_bins)
y_bins = np.arange(Y_GRID_RANGE[0], Y_GRID_RANGE[1], Y_GRID_RANGE[2])

def bev_tokens():
    """Return list of token strings for all grid positions."""
    tokens = []
    for xi in range(len(x_bins)):
        for yi in range(len(y_bins)):
            tokens.append(f"<bev_{xi}_{yi}>")
    return tokens

def clamp_index(value, bins):
    """Clip value to range and return nearest index."""
    idx = np.argmin(np.abs(bins - value))
    return int(idx)

def coord_to_token(x, y):
    """Convert real world coordinate to token."""
    xi = clamp_index(x, x_bins)
    yi = clamp_index(y, y_bins)
    return f"<bev_{xi}_{yi}>"

# ---------------------------------------------------------------------
# Decoding helpers – transform BEV tokens back to real‑world coordinates
# ---------------------------------------------------------------------

_BEV_REGEX = re.compile(r"<bev_(\d+)_(\d+)>")

def token_to_coord(token):
    """
    Convert a single BEV token (e.g. ``"<bev_12_7>"``) back to the (x, y)
    coordinate corresponding to the bin centres defined by *x_bins* and
    *y_bins*.

    Parameters
    ----------
    token : str
        Token string produced by :func:`coord_to_token`.

    Returns
    -------
    tuple[float, float]
        The (x, y) coordinate in metres.
    """
    m = _BEV_REGEX.fullmatch(token)
    if m is None:
        raise ValueError(f"Invalid BEV token: {token}")
    xi, yi = map(int, m.groups())
    # Guard against out‑of‑range indices
    if xi >= len(x_bins) or yi >= len(y_bins):
        raise IndexError(f"BEV index out of range in token {token}")
    return float(x_bins[xi]), float(y_bins[yi])

def decode_bev_tokens(text: str) -> list[tuple[float, float]]:
    """
    Extract all BEV tokens from *text* and decode them to (x, y) coordinates.
    If no BEV tokens are found, return an empty list [].
    This function does not return the modified string, only extracts the
    coordinates for consistent handling by higher layers.
    """
    coords: list[tuple[float, float]] = []
    for m in _BEV_REGEX.finditer(text):
        xi, yi = map(int, m.groups())
        # 越界保护
        if xi >= len(x_bins) or yi >= len(y_bins):
            raise IndexError(f"BEV index out of range: <bev_{xi}_{yi}>")
        coords.append( (float(x_bins[xi]), float(y_bins[yi])) )
    return coords
