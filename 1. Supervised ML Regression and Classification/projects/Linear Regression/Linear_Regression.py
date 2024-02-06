import numpy as np

class LinearRegression:

    coef_ = None
    bias_ = None
    n_iters_ = None

    def __init__(self, learning_rate=0.001, max_iters=1000, tol=1e-6, lambda_=0.01, random_state=None):
        """
        Initialize the Linear Regression model.

        Parameters:
        - learning_rate: The learning rate for gradient descent.
        - max_iters: Maximum number of iterations for gradient descent.
        - tol: Tolerance for convergence.
        - lambda_: Regularization parameter for L2 regularization.
        - random_state: Seed for random number generation.
        """
        self.lr = learning_rate
        self.max_iters = max_iters
        self.tol = tol
        self.lambda_ = lambda_
        self.random_state = random_state
        self.weights = None
        self.bias = None

    def __gradient_descent(self, m, X, y):
        """
        Perform batch gradient descent to minimize the cost function.       
        """
        # Initialize previous cost to infinity
        prev_cost = float('inf')

        # Perform gradient descent for max_iters iterations
        for _ in range(self.max_iters):
            # Calculate gradient
            dj_dw, dj_db = self.__calc_gradient(X, y, m)

            # Update weights and bias
            self.weights -= self.lr * dj_dw
            self.bias -= self.lr * dj_db

            # Calculate current cost
            current_cost = self.__calc_cost(X, y, m)

            # Check if it has converged (Eplison check)
            if abs(prev_cost - current_cost) <= self.tol:
                self.n_iters_ = _ + 1
                self.coef_ = self.weights
                self.bias_ = self.bias
                return
            
            prev_cost = current_cost

        self.n_iters_ = self.max_iters
        self.coef_ = self.weights
        self.bias_ = self.bias

    def __calc_gradient(self, X, y, m):
        """
        Calculate the gradient of the cost function.
        """
        # Calculate the predicted value
        y_approx = self.predict(X)

        # Calculate the gradient for the weights and bias
        dj_dw = (2/m) * np.dot(X.T, (y_approx - y)) + 2 * self.lambda_ * self.weights
        dj_db = (2/m) * np.sum(y_approx - y)
        return dj_dw, dj_db
    
    def __calc_cost(self, X, y, m):
        """
        Calculate the cost function.
        """
        # Calculate the predicted value
        y_approx = self.predict(X)

        # regularization term
        reg_term = (self.lambda_/2*m) * np.sum(self.weights**2)

        # Calculate the cost (mean squared error)
        cost = (1/m) * np.sum((y_approx - y)**2)

        return cost + reg_term

    def fit(self, X, y):
        """
        Fits the linear regression model to the training data.

        Parameters:
        - X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data.
        - y : ndarray of shape (n_samples,)
            Target values.

        Returns:
        - None
        """
        
        # init parameters
        if self.random_state is not None:
            np.random.seed(self.random_state)

        m, n_features = X.shape
        self.bias = 0
        self.weights = np.random.randn(n_features)

        # Perform gradient descent to update weights and bias
        self.__gradient_descent(m, X, y)

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        - X: Input features.

        Returns:
        - y_approx: Predicted values.
        """
        y_approx = np.dot(X, self.weights) + self.bias
        return y_approx