def _autoconfigure_hyperparameters(examples):
    return {
        "num_node_features": examples[0].node_features.shape[0],
        "num_edge_features": examples[0].fc_matrix.shape[2],
        "hidden_channels": [examples[0].node_features.shape[0] * 2],
        "use_pooling": False,
        "dropout_prob": 0.0,
        "global_pooling_mode": "mean",
        "conv_activation_func": 
    }


def create_gnn(examples, hyperparameters={}):
    hyperparameters = {
        **_autoconfigure_hyperparameters(examples),
        **hyperparameters
    }
