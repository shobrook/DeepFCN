from outgraph import Graph


class Example(object):
    def __init__(self, node_features, fc_matrix, y):
        # TODO: Add data shape validations
        self.node_features = node_features
        self.fc_matrix, self.y = fc_matrix, y

    def first_value_fc_matrix(self):
        return self.fc_matrix[:, :, :1]

    def to_data_obj(self):
        pass # TODO

    def to_outgraph(self):
        return Graph(self.node_features, self.first_value_fc_matrix())
