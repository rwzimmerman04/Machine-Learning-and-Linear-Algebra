"""Vector utilities built from first principles.

Notes
-----
- A vector is any sequence of numbers (list, tuple, or 1-D NumPy array).
- These functions use plain Python loops on purpose, to show how each
  operation works. Verify against np.linalg.norm.

Functions
---------
- calc_euclidean(vector): Euclidean magnitude (length) of a vector.
- normalize(vector): Scale a vector to unit length.
"""

import math


def calc_euclidean(vector):
    """Calculates the Euclidean magnitude of a vector."""

    magnitude_squared = 0

    for element in vector:
        magnitude_squared += element ** 2

    return math.sqrt(magnitude_squared)


def normalize(vector):
    """Returns a normalized vector with unit length."""

    magnitude = calc_euclidean(vector)

    if magnitude == 0:
        raise ValueError("Cannot normalize a zero vector.")

    return [element / magnitude for element in vector]