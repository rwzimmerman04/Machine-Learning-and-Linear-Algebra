import numpy as np


class Perceptron:
    """A basic binary perceptron classifier."""

    def __init__(self, learning_rate=0.01, n_epochs=100, random_state=None):
        """Initialize the perceptron.

        Args:
            learning_rate: Step size used to update the weights.
            n_epochs: Number of passes over the training data.
            random_state: Optional seed for reproducible weight initialization.
        """
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.weights = None
        self.bias = 0.0
        self.errors_ = []


