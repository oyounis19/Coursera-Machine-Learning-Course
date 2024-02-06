import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def __predict(self, inputs):
        node = self.tree
        while 'index' in node:
            if inputs[node['index']] < node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['predicted_class']

    def __best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None

        best_gini = float('inf')
        best_index, best_threshold = None, None

        # Loop through all features to find the best feature to split on
        for i in range(n):
            thresholds, classes = zip(*sorted(zip(X[:, i], y)))
            classes = np.array(classes)
            for j in range(1, m):
                if classes[j] != classes[j - 1]:
                    threshold = (thresholds[j] + thresholds[j - 1]) / 2
                    y_left = y[X[:, i] < threshold]
                    y_right = y[X[:, i] >= threshold]
                    gini = (len(y_left) * self.__gini_impurity(y_left) + len(y_right) * self.__gini_impurity(y_right)) / m
                    if gini < best_gini:
                        best_gini = gini
                        best_index = i
                        best_threshold = threshold

        return best_index, best_threshold

    def __gini_impurity(self, y):
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    def __grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in np.unique(y)]
        predicted_class = np.argmax(num_samples_per_class)
        node = {'predicted_class': predicted_class}

        if depth < self.max_depth:
            index, threshold = self.__best_split(X, y)
            if index is not None:
                indices_left = X[:, index] < threshold
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node['index'] = index
                node['threshold'] = threshold
                node['left'] = self.__grow_tree(X_left, y_left, depth + 1)
                node['right'] = self.__grow_tree(X_right, y_right, depth + 1)
        return node

    def fit(self, X, y):
        self.tree = self.__grow_tree(X, y)

    def predict(self, X):
        return np.array([self.__predict(inputs) for inputs in X])

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
