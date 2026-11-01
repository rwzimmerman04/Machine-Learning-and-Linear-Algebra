


def calc_trace(matrix):
    """Computes the trace of a square matrix."""

    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("Matrix must be a square and non-empty.")

    trace = 0

    for i in range(len(matrix)):
        trace += matrix[i][i]

    return trace