# Third Party
import numpy as np
from nilearn.connectome import ConnectivityMeasure
from pyts.metrics import dtw as _extract_dtw
from statsmodels.tsa.stattools import grangercausalitytests

FC_FEATURES = ["correlation", "dtw", "granger_causality"]


#########
# HELPERS
#########


def _extract_correlation(signal_i, signal_j):
    return np.correlation(signal_i, signal_j)[0]


def _extract_granger_causality(signal_i, signal_j):
    # Tests if signal_j predicts signal_i
    granger_test_result = grangercausalitytests(
        np.array([signal_i, signal_j]).T,
        maxlag=3,
        verbose=False
    )

    f_test_results = []
    for lag, results in granger_test_result.items():
        test_stats, ols_results = results
        F, p_val, df_denom, df_num = test_stats["ssr_ftest"]
        # chi2, p_val, df_num = test_stats["ssr_chi2test"]

        f_test_results.append(p_val)

    return min(f_test_results)


def _generate_matrix_indices(num_rois, num_features):
    for i in range(num_rois):
        for j in range(i):
            for k in range(num_features):
                yield i, j, k


######
# MAIN
######


def extract_fcn(signals, feature_names=["correlation"]):
    """
    Extracts a functional connectivity network (FCN) from BOLD signals. The FCN

    Parameters
    ----------
    signals : np.ndarray
        Array of BOLD signals; shape = [num_rois, time_series_len]
    feature_names : list
        Names of functional connectivity features to extract

    Returns
    -------
    numpy.ndarray
        FC matrix with shape [num_nodes, num_nodes, num_features]; multi-edge
        adjacency matrix representation of a FCN
    """

    num_rois, num_features = len(signals), len(feature_names)
    fc_matrix = np.empty([num_rois, num_rois, num_features])

    for i, j, k in _generate_matrix_indices(num_rois, num_features):
        signal_i, signal_j = signals[i], signals[j]
        feature_name = feature_names[k]

        if feature_name == "correlation":
            feature = _extract_correlation(signal_i, signal_j)
        elif feature_name == "dtw":
            feature = _extract_dtw(signal_i, signal_j)
        elif feature_name == "granger_causality":
            feature = _extract_granger_causality(signal_i, signal_j)

        fc_matrix[i][j][k] = feature
        fc_matrix[j][i][k] = feature

    return fc_matrix
