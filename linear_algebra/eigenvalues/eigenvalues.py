import math
import numpy as np

from utilities.vectors import calc_euclidean
from utilities.matrices import calc_trace
from eigenvectors import eigenvector_2x2

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
    
    return lambda1, lambda2


# Define and print out the matrix
my_matrix = [[0.61655556, 0.61544444], [0.61544444, 0.71655556]]
print(f"Matrix: {my_matrix}")

# Calculate and print out the Eigenvalues
print("")
result = eigenvalues_2x2(my_matrix)
print(f"Eigenvalues: {np.round(result, 8)}")

# Print out the Eigenvector of each Eigenvalue
for e in result:
    vector = eigenvector_2x2(my_matrix, e)
    print(f"Eigenvector for {np.round(e, 8)}:", np.round(vector, 8))
    print(f"Length of vector: {calc_euclidean(vector)}")


# Calculate the Eigenvalues and Eigenvectors with numpy to verify the results
print("")
print("Calculate the Eigenvalues and Eigenvectors with numpy.linalg.eigh to verify!")

nle_eigenvalues, nle_eigenvectors = np.linalg.eig(my_matrix)

print(f"Numpy Eigenvalues: {nle_eigenvalues}")
print(f"Numpy Eigenvectors: {nle_eigenvectors}")
