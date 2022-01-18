from multiprocessing import cpu_count


class Example(object):
    def __init__(self, node_features, fc_matrix, y):
        # TODO: Add data shape validations
        self.node_features = node_features
        self.fc_matrix, self.y = fc_matrix, y

    def to_data_obj(self):
        pass # TODO


def create_examples(images, label, roi_masker, fc_measures=["correlation"],
                    node_features=["mean"], n_jobs=cpu_count()):
    pass
