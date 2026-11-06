"""Matrix utilities built from first principles.

Notes
-----
- A matrix is a sequence of rows, such as a list of lists or a 2-D NumPy
  array, where matrix[i][j] is the entry in row i, column j.
- These functions use plain Python loops on purpose, to show how each
  operation works. Verify against np.trace.

Functions
---------
- calc_trace(matrix): Sum of the diagonal entries of a square matrix.
"""


def calc_trace(matrix):
    """Computes the trace of a square matrix."""

    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("Matrix must be a square and non-empty.")

    trace = 0

    for i in range(len(matrix)):
        trace += matrix[i][i]

    return trace