# Standard Library
from multiprocessing import Pool as ProcessPool, cpu_count

# Local
from data.extract_signals import extract_signals
from data.extract_node_features import extract_node_features
from data.extract_fcn import extract_fcn
from data.Example import Example


def _concurrent_exec(func, iterable, n_processes=-1):
    """
    Executes a function on a list of inputs in parallel.

    Parameters
    ----------
    func : callable
        function to execute
    iterable : array
        list of inputs to map the function to
    n_processes : int or None
        number of worker processes to use; if -1, then the number returned by
        os.cpu_count() is used

    Returns
    -------
    list : function outputs corresponding to each input
    """

    pool = ProcessPool(processes=None if n_processes == -1 else n_processes)
    map_of_items = pool.map(func, iterable)
    pool.close()
    pool.join()

    return map_of_items


def create_examples(images, label, roi_masker, fc_features=["correlation"],
                    node_features=["mean"], n_jobs=-1):
    """
    Parameters
    ----------
    images :
    label :
    roi_masker :
    fc_features :
    node_features :
    n_jobs :

    Returns
    -------
    list :
    """

    def _create_example(image):
        bold_signals = extract_signals(image, roi_masker)
        node_features = extract_node_features(bold_signals, node_features)
        fc_matrix = extract_fcn(bold_signals, fc_features)

        return Example(node_features, fc_matrix, label)

    examples = _concurrent_exec(_create_example, images, n_jobs)
    return list(examples)
