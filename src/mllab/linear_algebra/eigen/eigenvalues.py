import math
import numpy as np
import argparse

from mllab.utilities.vectors import calc_euclidean
from mllab.utilities.matrices import calc_trace
from mllab.linear_algebra.eigen.eigenvectors import eigenvector_2x2

def eigenvalues_2x2(matrix):
    """Computes the eigenvalues of a given 2x2 matrix."""
    a, b = matrix[0]
    c, d = matrix[1]
    
    # Calculate the trace and determinant
    trace = calc_trace(matrix)
    det = a*d - b*c
    
    # Calculate the discriminant for retrieving the eigenvalues.
    discriminant = trace**2 - 4*det
    
    if discriminant < 0:
        # Discriminant is negative, this will result in Complex Eigenvalues
        real_part = trace / 2
        imag_part = math.sqrt(-discriminant) / 2	

	# Report complex numbers for the Eigenvalues
        lambda1 = complex(real_part, imag_part)
        lambda2 = complex(real_part, -imag_part)

        print(f"Check the input matrix and work done! Eigenvalues are complex!: [{lambda1}, {lambda2}]")
    else:
	# Discriminant is Positive -> Real Eigenvalues
        lambda1 = (trace + math.sqrt(discriminant)) / 2
        lambda2 = (trace - math.sqrt(discriminant)) / 2
    
    return [float(lambda1), float(lambda2)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('matrix')

    args = parser.parse_args()

    my_matrix = args.matrix

    eigenvalues_2x2(my_matrix)
