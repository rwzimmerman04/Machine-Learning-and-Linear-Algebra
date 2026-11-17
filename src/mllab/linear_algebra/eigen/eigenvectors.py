from mllab.utilities.vectors import normalize

def eigenvector_2x2(matrix, eigenvalue):
    """Computes the Egienvectors of a given matrix."""
    a, b = matrix[0]
    c, d = matrix[1]
    
    # Calculate the diagonal values of A - lambda * I
    a_minus_lambda = a - eigenvalue
    d_minus_lambda = d - eigenvalue
    
    # Attempt to use the first row: a_minus_lambda*v1 + b*v2 = 0
    if abs(b) > 0:  # b is not zero
        v1 = 1
        v2 = -(a_minus_lambda / b) * v1
    
    # If b is zero, use the second row: c*v1 + d_minus_lambda*v2 = 0
    elif abs(c) > 0:  # c is not zero
        v2 = 1
        v1 = -(d_minus_lambda / c) * v2


    return normalize([float(v1), float(v2)])