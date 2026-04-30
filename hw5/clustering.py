import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans


RANDOM_STATE = 42

N_CLUSTERS_LIST = [35, 40, 45]
N_COMPONENTS_LIST = [100, 200, 300, 500, 1000]


def load_data(path: str):
    """Загрузка и базовая очистка данных."""
    train = sparse.load_npz(path).astype(np.float32)

    X = train.copy()
    X.data = np.nan_to_num(
        X.data,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )
    X.eliminate_zeros()

    return X


def preprocess(X):
    return normalize(X, norm="l2", axis=1)


def save_submission(labels: np.ndarray, filename: str):
    submission = pd.DataFrame(
        {
            "ID": np.arange(len(labels)),
            "TARGET": labels.astype(int),
        }
    )

    submission.to_csv(filename, index=False)
    print(f"Saved: {filename}")


def run_pipeline(X):
    for n_components in N_COMPONENTS_LIST:
        print(f"\nSVD components: {n_components}")

        svd = TruncatedSVD(
            n_components=n_components,
            random_state=RANDOM_STATE,
        )

        X_svd = svd.fit_transform(X)

        X_svd = np.nan_to_num(
            X_svd,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        explained = svd.explained_variance_ratio_.sum()
        print(f"Explained variance: {explained:.5f}")

        X_svd = normalize(X_svd, norm="l2", axis=1)

        for n_clusters in N_CLUSTERS_LIST:
            print(
                f"Training KMeans: "
                f"SVD={n_components}, k={n_clusters}"
            )

            model = KMeans(
                n_clusters=n_clusters,
                init="k-means++",
                n_init=100,
                max_iter=500,
                random_state=RANDOM_STATE,
                verbose=0,
            )

            labels = model.fit_predict(X_svd)

            filename = (
                f"submission_kmeanspp_"
                f"svd{n_components}_k{n_clusters}.csv"
            )

            save_submission(labels, filename)


if __name__ == "__main__":
    X = load_data("train.npz")
    X = preprocess(X)

    run_pipeline(X)
