# ML Learning Lab

A collection of Python implementations, examples, and notes covering the mathematics and concepts behind machine learning.

The goal of this repository is to build a practical understanding of mathematical and machine learning concepts by implementing them in Python. Each topic includes explanations of the underlying mathematics and example implementations.

Rather than relying entirely on existing libraries, this project focuses on understanding how the underlying operations work and how they contribute to larger algorithms.

## Topics (Planned)

### Linear Algebra
- Vectors and vector operations
- Matrices and matrix operations
- Eigenvalues and eigenvectors
- Matrix transformations and decompositions

### Probability and Statistics
- Probability fundamentals
- Conditional probability and independent events
- Bayes' theorem
- Probability distributions
- Mean, variance, and standard deviation

### Machine Learning
- Mathematical functions used in ML
- Classification algorithms
- Naive Bayes
- Model evaluation and performance metrics
- Optimization and gradient descent
- Loss functions and learning rate schedules

## Repository Structure

```text
ml-learning-lab/
│
├── README.md
├── LICENSE
├── LICENSE-DOCS
│
├── utilities/
│   ├── vector_operations.py
│   └── matrix_operations.py
│
├── linear_algebra/
│   ├── README.md
│   ├── matrices/
│   ├── vectors/
│   └── eigenvalues/
│
├── probability/
│   ├── README.md
│   ├── fundamentals/
│   ├── distributions/
│   └── statistics/
│
└── machine_learning/
    ├── README.md
    ├── functions/
    ├── algorithms/
    ├── evaluation/
    └── optimization/
```

The repository will grow incrementally as new concepts are covered and implemented.

## Implementation Approach

Each topic aims to include:

1. **Concept overview:** An explanation of the concept and its purpose.
2. **Mathematical foundation:** Relevant formulas, variables, and assumptions.
3. **Python implementation:** An implementation of the concept.
4. **Examples:** Small numerical examples demonstrating how it works.

## Shared Utilities

Common operations are maintained separately from individual topic implementations. This allows functions to be reused across different areas of the repository without duplicating code.

For example, vector and matrix operations implemented in the utilities directory can be imported into linear algebra, probability, and machine learning examples.

## Tools and Libraries

- Python
- NumPy
- Matplotlib
