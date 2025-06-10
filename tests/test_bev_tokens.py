#!/usr/bin/env python
"""Basic checks for BEV token utilities."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train.llava.bev_utils import bev_tokens, coord_to_token, x_bins, y_bins


def main():
    tokens = bev_tokens()
    expected = len(x_bins) * len(y_bins)
    assert len(tokens) == expected, f"Expected {expected} tokens, got {len(tokens)}"
    assert len(tokens) == len(set(tokens)), "Tokens are not unique"

    # sample coordinates
    for x, y in [(0, 0), (-39.5, -59.9), (39.9, 59.8)]:
        token = coord_to_token(x, y)
        assert token in tokens, f"Token {token} not in token list"


if __name__ == "__main__":
    main()
    print("All BEV token tests passed.")
