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

    