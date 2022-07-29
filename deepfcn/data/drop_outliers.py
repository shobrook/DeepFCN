from outgraph import detect_outliers


def drop_outliers(examples, cutoff=0.05):
    """
    Converts FCNs into vectors and uses the Mahalanobis distance of each vector
    from the distribution to detect and drop outliers.

    Parameters
    ----------
    examples : list
        list of Example objects representing the dataset to detect outliers in
    cutoff : float
        p-value threshold to use in determining which examples are outliers

    Returns
    -------
    list : example set with outliers removed
    """

    graphs = [example.to_outgraph() for example in examples]
    _, indices = detect_outliers(graphs, method=3, p_value=cutoff)

    return [examples[i] for i in indices]
