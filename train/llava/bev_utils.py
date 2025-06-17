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

def decode_bev_tokens(text):
    """
    Replace every BEV token in *text* with its decoded ``(x, y)`` string
    representation, keeping other text intact.

    Example
    -------
    >>> decode_bev_tokens('Car at <bev_12_7> approaching <bev_15_7>')
    'Car at (x=??, y=??) approaching (x=??, y=??)'
    """
    def _sub(match):
        x, y = token_to_coord(match.group(0))
        return f"(x={x:.2f}, y={y:.2f})"

    return _BEV_REGEX.sub(_sub, text)
