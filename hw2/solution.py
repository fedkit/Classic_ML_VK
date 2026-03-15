import numpy as np


class LinearRegression:
    def __init__(
        self,
        *,
        penalty="l2",
        alpha=0.0001,
        max_iter=1000,
        tol=0.001,
        random_state=None,
        eta0=0.01,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        shuffle=True,
        batch_size=32,
    ):
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.eta0 = eta0
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.shuffle = shuffle
        self.batch_size = batch_size

        self._count_features = None
        self._count_labels = None

        self._coef = None
        self._intercept = None

    def get_penalty_grad(self):
        if self.penalty == "l1":
            return self.alpha * np.sign(self._coef)

        if self.penalty == "l2":
            return 2 * self.alpha * self._coef

        return 0 * self._coef

    def _training_model(self, x_train, y_train, x_val=None, y_val=None):
        count_train = x_train.shape[0]
        count_iter_no_change = 0
        best_loss = np.inf
        count_iter = 0

        while count_iter < self.max_iter and \
            count_iter_no_change < self.n_iter_no_change:
            
            count_iter += 1

            for i in range(0, count_train, self.batch_size):
                end_batch = i + self.batch_size
                x_batch = x_train[i:end_batch]
                y_batch = y_train[i:end_batch]

                pred = np.dot(x_batch, self._coef) + self._intercept

                w_grad = (
                    2 * np.dot(x_batch.T, pred - y_batch) / x_batch.shape[0]
                    + self.get_penalty_grad()
                )
                self._coef -= self.eta0 * w_grad

                w0_grad = 2 * np.sum(pred - y_batch) / x_batch.shape[0]
                self._intercept -= self.eta0 * w0_grad

            if self.early_stopping and x_val is not None:
                val_pred = np.dot(x_val, self._coef) + self._intercept
                val_loss = np.mean((val_pred - y_val) ** 2)

                if best_loss - val_loss > self.tol:
                    best_loss = val_loss
                    count_iter_no_change = 0
                else:
                    count_iter_no_change += 1
            else:
                count_iter_no_change = 0

    def fit(self, x, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self._count_features = x.shape[1]
        self._count_labels = x.shape[0]
        self._coef = np.random.rand(self._count_features)
        self._intercept = 0

        indices = np.arange(self._count_labels)
        if self.shuffle:
            np.random.shuffle(indices)

        if self.early_stopping:
            val_threshold = int(self._count_labels * (1 - self.validation_fraction))
            x_dop, y_dop = x[indices], y[indices]
            x_train, x_val = x_dop[:val_threshold], x_dop[val_threshold:]
            y_train, y_val = y_dop[:val_threshold], y_dop[val_threshold:]
            self._training_model(x_train, y_train, x_val, y_val)
        else:
            x_train, y_train = x[indices], y[indices]
            self._training_model(x_train, y_train)

    def predict(self, x):
        return np.dot(x, self._coef) + self._intercept

    @property
    def coef_(self):
        return self._coef

    @property
    def intercept_(self):
        return self._intercept

    @coef_.setter
    def coef_(self, value):
        self._coef = value

    @intercept_.setter
    def intercept_(self, value):
        self._intercept = value
