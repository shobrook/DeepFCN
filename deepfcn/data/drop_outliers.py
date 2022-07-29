from outgraph import detect_outliers


def drop_outliers(examples, cutoff=0.05):
    """
    """

    graphs = [example.to_outgraph() for example in examples]
    _, indices = detect_outliers(graphs, method=3, p_value=cutoff)

    return [examples[i] for i in indices]
