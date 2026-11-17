# Eigenvalues and Eigenvectors

## What Are Eigenvalues and Eigenvectors?

Eigenvalues and eigenvectors describe how a linear transformation affects specific directions in a vector space.

When a matrix transforms a vector, it generally changes both its magnitude and direction. However, some nonzero vectors retain their original direction after the transformation. These vectors are called **eigenvectors**, and the corresponding **eigenvalues** describe how much those vectors are scaled.

The relationship between a matrix, an eigenvector, and its eigenvalue is:

$$Av = \lambda v$$

Where:

- $A$ is a square matrix representing a linear transformation.
- $v$ is a nonzero eigenvector.
- $\lambda$ is the corresponding eigenvalue.

A positive eigenvalue scales the eigenvector without reversing its direction, a negative eigenvalue reverses its direction, and an eigenvalue of zero means the vector is transformed into the zero vector.

## Calculating Eigenvalues and Eigenvectors by Hand

Consider the following 2x2 matrix:

$$
A =
\begin{bmatrix}
4 & 2 \\
1 & 3
\end{bmatrix}
$$

To find its eigenvalues, we need to determine the values of $\lambda$ where the matrix transforms a nonzero vector into a scalar multiple of itself.

Starting with:

$$Av = \lambda v$$

To express the scalar multiplication as a matrix operation, we will introduce the identity matrix $I$, which has the same dimensions as $A$.

$$Av = (\lambda I)v$$


> **Note:** The identity matrix is the matrix equivalent of multiplying by 1. This does not change the value of $\lambda$, it simply lets us represent $\lambda$ as a matrix.

Rearranging gives:

$$(A-\lambda I)v=0$$

For a nonzero eigenvector to exist, the resulting matrix must be singular. This allows us to solve for the eigenvalues using its determinant.

Once the eigenvalues are known, each can be substituted back into the equation to solve for its corresponding eigenvector.

## The Characteristic Equation

The equation:

$$(A-\lambda I)v=0$$

has a nonzero solution only when $A-\lambda I$ is singular. Therefore, its determinant must equal zero:

$$\det(A-\lambda I)=0$$

This is called the **characteristic equation**. Solving it gives the eigenvalues of the matrix.


### The 2x2 Case

For a matrix:

$$
A =
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
$$

Expanding the determinant in the characteristic equation gives:

$$
\lambda^2-\operatorname{tr}(A)\lambda+\det(A)=0
$$

This is a quadratic equation, which can be solved using the quadratic formula:

$$
\lambda =
\frac{\operatorname{tr}(A)\pm
\sqrt{\operatorname{tr}(A)^2-4\det(A)}}{2}
$$

The quadratic formula has the general form:

$$
x = \frac{-b\pm\sqrt{b^2-4ac}}{2a}
$$

The expression under the square root, $b^2-4ac$, is called the **discriminant**. It determines the nature and number of the equation's roots.

For the characteristic equation, the discriminant is:

$$
\Delta = \operatorname{tr}(A)^2-4\det(A)
$$

Its value determines the type of eigenvalues:

- **Positive discriminant:** Two distinct real eigenvalues.
- **Zero discriminant:** One repeated real eigenvalue.
- **Negative discriminant:** Two complex conjugate eigenvalues for a real matrix.

The discriminant is useful because it tells us what kind of eigenvalues to expect before calculating them. 

## Python Implementation

The repository contains educational Python implementations of the eigenvalue and eigenvector calculations for 2x2 matrices.

### Current Implementations

- `eigenvalues.py`: Calculates the eigenvalues of a 2x2 matrix using its trace, determinant, and the quadratic formula.
- `eigenvectors.py`: Calculates eigenvectors corresponding to supplied eigenvalues.

### Shared Utilities

The implementations reuse common operations from the repository's utilities directory:

- `calc_trace()`: Calculates the trace of a square matrix.
- `calc_euclidean()`: Calculates the Euclidean magnitude of a vector.
- `normalize()`: Returns a normalized vector.

## Future Work

- Handle special cases such as diagonal matrices and repeated eigenvalues.
- Improve eigenvector calculations to handle all 2x2 cases.
- Extend eigenvalue and eigenvector calculations to larger matrices.
- Investigate matrix diagonalization and its applications.