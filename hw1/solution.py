import numpy as np


def product_of_diagonal_elements_vectorized(matrix: np.array):
    return np.prod(np.diag(matrix)[np.nonzero(np.diag(matrix))])


def are_equal_multisets_vectorized(x: np.array, y: np.array):
    # в целом можно линейно решить данную задачу через dict,
    # но я не знаю как это сделать не используя for
    return np.array_equal(np.sort(x), np.sort(y))


def max_before_zero_vectorized(x: np.array):
    indices = np.where(x[:-1] == 0)[0] + 1
    return np.max(x[indices])


def add_weighted_channels_vectorized(image: np.array):
    return (
        image[..., 0] * 0.299
        + image[..., 1] * 0.587
        + image[..., 2] * 0.114
    )


def run_length_encoding_vectorized(x: np.array):
    changes = np.where(x[1:] != x[:-1])[0] + 1
    indices = np.append([0], np.append(changes, x.size))

    values = x[indices[:-1]]
    counts = np.diff(indices)

    return values, counts
