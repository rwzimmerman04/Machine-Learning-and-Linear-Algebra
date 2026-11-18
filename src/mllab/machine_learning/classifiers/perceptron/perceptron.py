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


    def net_input(self, x):
        """Compute the weighted sum of features plus the bias.

        Args:
            x: 1D array-like containing one sample's features.

        Returns:
            The perceptron's raw activation value.
        """
        return np.dot(x, self.weights) + self.bias


    def fit(self, X, y):
        """Train the perceptron on feature data and binary labels.

        Args:
            X: 2D array-like of shape (n_samples, n_features).
            y: 1D array-like of binary labels, conventionally 0 and 1.

        Returns:
            self: The fitted perceptron.
        """
        # Get the number of observations and features in the training data
        n_observations, n_features = X.shape

        # Start with zero-valued weights and bias.
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Train the model by passing over the data for each epoch.
        for _ in range(self.epochs):
            errors = 0

            # Process each observation and its corresponding label.
            for xi, yi in zip(X, y):
                # The linear output is the weighted sum of the features plus the bias.
                # It indicates which side of the decision boundary the observation falls on.
                linear_output = self.net_input(xi)

                # Convert the linear output into a binary prediction.
                yi_hat = 1 if linear_output >= 0 else 0

                # The update is zero for a correct prediction. For an incorrect
                # prediction, its sign indicates which direction to adjust the model.
                update = self.learning_rate * (yi - yi_hat)

                # Adjust the weights and bias using the prediction error.
                self.weights += update * xi
                self.bias += update

                # Count each incorrect prediction.
                errors += int(update != 0)

            # Store the number of errors made during this epoch.
            self.errors.append(errors)


   