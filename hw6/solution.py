import numpy as np
from sklearn.tree import DecisionTreeRegressor


class GBCustomRegressor:
    def __init__(
        self,
        *,
        learning_rate=0.1,
        n_estimators=100,
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        max_depth=3,
        random_state=None
    ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state

        self._estimators = []
        self._x = None
        self._y = None

    def fit(self, x, y):
        self._x = x
        self._y = y
        self._estimators = []

        current_prediction = np.full_like(y, np.mean(y))

        for i in range(self.n_estimators):
            residual = y - current_prediction
            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
            tree.fit(x, residual)

            current_prediction += self.learning_rate * tree.predict(x)
            self._estimators.append(tree)

        return self

    def predict(self, x):
        pred = np.full(x.shape[0], np.mean(self._y))
        for model in self._estimators:
            pred += self.learning_rate * model.predict(x)
        return pred

    @property
    def estimators_(self):
        return self._estimators


class GBCustomClassifier:
    def __init__(
        self,
        *,
        learning_rate=0.1,
        n_estimators=100,
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        max_depth=3,
        random_state=None
    ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state

        self._estimators = []
        self._x = None
        self._y = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, x, y):
        y = (y > 0).astype(float)
        self._x = x
        self._y = y
        current_prediction = np.full_like(y, np.log(np.mean(y) / (1 - np.mean(y))))

        self._estimators = []

        for i in range(self.n_estimators):
            prob = self._sigmoid(current_prediction)
            residual = y - prob

            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )

            tree.fit(x, residual)

            current_prediction += self.learning_rate * tree.predict(x)
            self._estimators.append(tree)

        return self

    def predict_proba(self, x):
        p = np.mean(self._y)
        prediction = np.full(x.shape[0], np.log(p / (1 - p)))

        for tree in self._estimators:
            prediction += self.learning_rate * tree.predict(x)

        prob = self._sigmoid(prediction)

        return np.vstack([1 - prob, prob]).T

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)

    @property
    def estimators_(self):
        return self._estimators
