import numpy as np
from math import floor


def _flatten_fc_matrix(fc_matrix):
    _fc_matrix = np.copy(fc_matrix)
    _fc_matrix = fc_matrix[:, :, :1]
    _fc_matrix[np.triu_indices(_fc_matrix.shape[0], k=0)] = np.nan
    _fc_matrix = -1 * np.absolute(_fc_matrix)

    return _fc_matrix.squeeze()


def _weak_edge_indices(fc_matrix, cutoff):
    n_edges = (fc_matrix.shape[0] * (fc_matrix.shape[0] - 1)) / 2
    n_drop = floor(n_edges * cutoff)
    n_nan = np.isnan(fc_matrix).sum()

    indices = np.argpartition(fc_matrix.ravel(), -n_drop-n_nan)[-n_drop-n_nan:-n_nan]
    indices = np.unravel_index(indices, fc_matrix.shape)

    return indices
    X[indices] = np.zeros(A[indices].shape)


def drop_weak_connections(examples, cutoff=0.1):
    """
    TODO

    Parameters
    ----------
    examples : list
        ...
    cutoff : float
        ...

    Returns
    -------
    list : examples with weak edges dropped in each of them
    """

    for example in examples:
        fc_matrix = _flatten_fc_matrix(example.fc_matrix)
        lt_indices = _weak_edge_indices(fc_matrix, cutoff)
        ut_indices = lt_indices[::-1]

        n_fc_features = example.fc_matrix.shape[-1]
        example.fc_matrix[lt_indices] = np.zeros(n_fc_features)
        example.fc_matrix[ut_indices] = np.zeros(n_fc_features)

    return examples
