import numpy as np

class LogisticRegression:

    coef_ = None
    bias_ = None
    n_iters_ = None

    def __init__(self, C=0.001, max_iter=100, threshold=0.5, penalty="l2", tol=1e-4, lambda_=0.01, random_state=None):
        if penalty not in ["l1", "l2"]:
            raise Exception('penalty must be in: "l2", "l1"')
        if random_state is not None:
            np.random.seed(random_state)

        self.lr = C
        self.max_iters = max_iter
        self.threshold = threshold
        self.penalty = penalty
        self.tol = tol
        self.lambda_ = lambda_
        self.weights = None
        self.bias = None

    def __l1(self):
        return np.sum(np.abs(self.weights)) 

    def __l2(self):
        return np.sum(self.weights**2)
    
    def __sigmoid(self, X):
        z = np.dot(X, self.weights) + self.bias
        return 1 / (1 + np.exp(-z))
    
    def __calc_cost(self, X, y):
        m, _ = X.shape
        y_approx = self.__sigmoid(X)
        cost = (1/m) * np.sum(-y*np.log(y_approx) - (1-y)*np.log(1-y_approx))
        # reg_term = (self.lambda_/2*m) * np.sum(self.weights**2)# before derivative
        # reg_term = (self.lambda_/m) * np.sum(self.weights)# after derivative
        reg_term = (self.lambda_/m) * self.__l1() if self.penalty is "l1" else (self.lambda_/m) * self.__l2()
        return cost

    def __gradient_descent(self, X, y):
        # Initialize previous cost to infinity
        prev_cost = float('inf')
        for _ in range(self.max_iters):
            dj_dw, dj_db = self.__calc_gradient(X, y)

            # Update weights and bias
            self.weights -= self.lr * dj_dw
            self.bias -= self.lr * dj_db

            # Calculate current cost
            current_cost = self.__calc_cost(X, y)

            # Check if it has converged (Epsilon check)
            if abs(prev_cost - current_cost) <= self.tol:
                self.n_iters_ = _ + 1
                self.coef_ = self.weights
                self.bias_ = self.bias
                return

            prev_cost = current_cost
        
        self.n_iters_ = self.max_iters
        self.coef_ = self.weights
        self.bias_ = self.bias

    def __calc_gradient(self, X, y):
        m, n = X.shape
        # Calculate the predicted value
        y_approx = self.__sigmoid(X)

        # dj_dw = 0
        # dj_db = 0
        # # Calculate the gradient for the weights and bias
        # for i in range(m):
        #     for j in range(n):
        #         dj_dw += (y_approx[i] - y[i]) * X[i, j]
        #     dj_dw = (dj_dw/m) + self.__l1() if self.penalty is "l1" else (dj_dw/m) * self.__l2()
        #     dj_db += (y_approx[i] - y[i])

        dj_dw = (1/m) * np.dot(X.T, (y_approx - y)) + (self.lambda_/m)
        dj_dw *= self.__l1() if self.penalty is "l1" else self.__l2()
        dj_db = (1/m) * np.sum(y_approx - y)

        return dj_dw, dj_db

    def fit(self, X, y):
        # init parameters
        m, n = X.shape
        self.bias = 0
        self.weights = np.random.randn(n)

        self.__gradient_descent(X, y)

    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(int)
    
    def predict_proba(self, X):
        return self.__sigmoid(X)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)