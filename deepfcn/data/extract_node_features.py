TIME_SERIES_NODE_FEATURES = ["entropy", "fractal_dim", "lyap_r", "dfa", "mean",
                             "median", "range", "std", "auto_corr", "auto_cov"]
NETWORK_NODE_FEATURES = ["weighted_degree", "clustering_coef",
                         "closeness_centrality", "betweenness_centrality"]
NODE_FEATURES = TIME_SERIES_NODE_FEATURES + NETWORK_NODE_FEATURES


def create_networkx_graph(signals):
    fc_matrix = extract_fc_matrix(signals)
    return nx.from_numpy_matrix(np.matrix(fc_matrix))


def extract_node_network_feats(G, example):
    # NOTE: Dijkstra's algorithm is used for computing shortest path lengths for
    # the closeness and betweenness centrality. So, weights have to be
    # recalculated to represent "distances" instead of correlations.
    H = G.copy()
    for i, j, data in H.edges(data=True):
        data["weight"] = 1 / abs(data["weight"])

    # Creates node attribute lookup tables
    get_clustering_coef = nx.clustering(G, weight="weight")
    get_degree_centrality = nx.degree_centrality(G)
    get_closeness_centrality = nx.closeness_centrality(H, distance="weight")
    get_betweenness_centrality = nx.betweenness_centrality(H, weight="weight")

    for i in range(G.number_of_nodes()):
        example["node_feats"][i].update({
            "degree": G.degree[i],
            "clustering_coef": get_clustering_coef[i],
            "degree_centrality": get_degree_centrality[i],
            "closeness_centrality": get_closeness_centrality[i],
            "betweenness_centrality": get_betweenness_centrality[i]
        })

def right_shift_time_series(time_series, shift_by):
    rs_time_series = []
    for i in range(len(time_series) - 1):
        rs_time_series.append(time_series[i + 1])

    return rs_time_series


def calculate_auto_corr_and_cov(time_series):
    rs_time_series = right_shift_time_series(time_series, shift_by=1)
    slope, intercept, auto_corr, p_val, std_err = linregress(
        time_series[:-1],
        rs_time_series
    )
    auto_cov = slope * variance(time_series)

    return auto_corr, auto_cov

EXTRACT_FEATURE = {
    "lyap_r": nolds.lyap_r,
    "hurst_rs": nolds.hurst_rs,

}

def extract_node_features(signals, features):
    num_rois, num_features = signals.shape[0], len(features)
    node_features = np.empty([num_rois, num_features])
    for i in range(num_rois):
        signal = signals[i]






auto_corr, auto_cov = calculate_auto_corr_and_cov(node_ts)
        example["node_feats"][i].update({
            # "approx_entropy": nolds.sampen(node_ts), # BUG: Produces inf values
            # "fractal_dim": nolds.corr_dim(node_ts, emb_dim=10), # BUG: Always produces the same value
            "lyap_r": nolds.lyap_r(node_ts),
            "hurst_rs": nolds.hurst_rs(node_ts),
            "dfa": nolds.dfa(node_ts),
            "mean": mean(node_ts),
            "median": median(node_ts),
            "range": max(node_ts) - min(node_ts),
            "std": stdev(node_ts),
            "auto_corr": auto_corr,
            "auto_cov": auto_cov,
            "skew": skew(node_ts),
            "kurtosis": kurtosis(node_ts)
        })
