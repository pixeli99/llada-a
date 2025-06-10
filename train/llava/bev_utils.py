import numpy as np

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
