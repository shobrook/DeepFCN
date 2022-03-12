# Standard Library
from statistics import mean, median, stdev, variance

# Third Party
import nolds
import numpy as np
import networkx as nx
from scipy.stats import kurtosis, skew, linregress

# Local
from data.extract_fcn import extract_fcn


######################
# TIME SERIES FEATURES
######################


TIME_SERIES_NODE_FEATURES = ["entropy", "fractal_dim", "lyap_r", "dfa", "mean",
                             "median", "range", "std", "auto_corr", "auto_cov"]
EXTRACT_TS_FEATURE = {
    # "approx_entropy": nolds.sampen, # BUG: Produces inf values
    # "fractal_dim": lambda node_ts: nolds.corr_dim(node_ts, emb_dim=10), # BUG: Always produces the same value
    "lyap_r": nolds.lyap_r,
    "hurst_rs": nolds.hurst_rs,
    "dfa": nolds.dfa,
    "mean": mean,
    "median": median,
    "range": lambda signal: max(signal) - min(signal),
    "std": stdev,
    "skew": skew,
    "kurtosis": kurtosis
}


def _right_shift_time_series(time_series, shift_by):
    rs_time_series = []
    for i in range(len(time_series) - 1):
        rs_time_series.append(time_series[i + 1])

    return rs_time_series


def _calculate_auto_corr_cov(time_series):
    rs_time_series = _right_shift_time_series(time_series, shift_by=1)
    slope, intercept, auto_corr, p_val, std_err = linregress(
        time_series[:-1],
        rs_time_series
    )
    auto_cov = slope * variance(time_series)

    return auto_corr, auto_cov


def _extract_time_series_features(signals, feature_names):
    feature_names = [f for f in feature_names if f in TIME_SERIES_NODE_FEATURES]
    num_rois, num_features = len(signals), len(feature_names)
    feature_matrix = np.empty([num_rois, num_features])

    if not feature_names:
        return feature_matrix

    for i in range(num_rois):
        for j, feature_name in enumerate(feature_names):
            if feature_name == "auto_corr":
                feature_matrix[i][j], _ = _calculate_auto_corr_cov(signals[i])
            elif feature_name == "auto_cov":
                _, feature_matrix[i][j] = _calculate_auto_corr_cov(signals[i])
            else:
                extract_feature = EXTRACT_TS_FEATURE[feature_name]
                feature_matrix[i][j] = extract_feature(signals[i])

    return feature_matrix


##################
# NETWORK FEATURES
##################


NETWORK_NODE_FEATURES = ["weighted_degree", "clustering_coef",
                         "closeness_centrality", "betweenness_centrality"]
EXTRACT_NETWORK_FEATURE = {
    "weighted_degree": lambda G, i: G.degree[i],
    "clustering_coef": lambda G, i: nx.clustering(G, weight="weight")[i],
    "degree_centrality": lambda G, i: nx.degree_centrality(G)[i],
    "closeness_centrality": lambda H, i: nx.closeness_centrality(H, distance="weight")[i],
    "betweenness_centrality": lambda H, i: nx.betweenness_centrality(H, distance="weight")[i]
}


def _create_networkx_graph(signals):
    fc_matrix = np.squeeze(extract_fcn(signals))
    return nx.from_numpy_matrix(np.matrix(fc_matrix))


def _create_reciprical_graph(G):
    # NOTE: Dijkstra's algorithm is used for computing shortest path lengths for
    # the closeness and betweenness centrality. So, weights have to be
    # recalculated to represent "distances" instead of correlations.
    H = G.copy()
    for i, j, data in H.edges(data=True):
        data["weight"] = 1 / abs(data["weight"])

    return H


def _extract_network_features(signals, feature_names):
    feature_names = [f for f in feature_names if f in NETWORK_NODE_FEATURES]
    num_rois, num_features = len(signals), len(feature_names)
    feature_matrix = np.empty([num_rois, num_features])

    if not feature_names:
        return feature_matrix

    G = _create_networkx_graph(signals)
    H = _create_reciprical_graph(G)

    for i in range(num_rois):
        for j, feature_name in enumerate(feature_names):
            extract_feature = EXTRACT_NETWORK_FEATURE[feature_name]
            if feature_name in ("closeness_centrality", "betweenness_centrality"):
                feature_matrix[i][j] = extract_feature(H, i)
            else:
                feature_matrix[i][j] = extract_feature(G, i)

    return feature_matrix


######
# MAIN
######


def extract_node_features(signals, feature_names=["mean"]):
    """
    Extracts node (ROI) features from BOLD signals. Features can either be
    calculated from the node's time series, or from the node's graph theoretic
    properties in a FCN.

    Parameters
    ----------
    signals : numpy.ndarray
        Array of BOLD signals; shape = [num_rois, time_series_len]
    feature_names : list
        Names of node features to extract

    Returns
    -------
    numpy.ndarray
        Array of ROI/node features; shape = [num_rois, num_features]
    """

    ts_features = _extract_time_series_features(signals, feature_names)
    network_features = _extract_network_features(signals, feature_names)

    return np.concatenate((ts_features, network_features), axis=1)
