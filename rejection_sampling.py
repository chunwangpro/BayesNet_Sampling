import numpy as np

from utils import *


def get_prior_sample():
    """Return one sample for each variable based on their CPTs.

    Returns:
    --------
    dict : A dictionary where the keys are node names and the values are sampled booleans.
    """
    sample = {}
    for k in topological():
        node = Node.nodes[k]
        if not node.parents:
            # for root node
            sample[k] = random.random() < node.prob
        else:
            # for non-root node
            parent_list = tuple(sample[p] for p in node.parents)
            sample[k] = node.sample(parent_list)
    return sample


def rejection_sampling(query, evidence, totalSamples=10000):
    """Perform rejection sampling to estimate the normalized probability distribution of the query.

    Parameters:
    -----------
    query : str
        The name of the query node.
    evidence : dict
        A dictionary mapping node names (strings) to boolean values.
    totalSamples : int
        The total number of samples to generate.

    Returns:
    --------
    float : The estimated probability.
    """
    ind = topological().index(query)
    results = []
    for _ in range(totalSamples):
        # First get one sample using prior sampling
        sample = get_prior_sample()
        # Only accept samples that are consistent with the evidence
        for k, v in evidence.items():
            if sample[k] != v:
                break
        else:
            results.append(list(sample.values()))
    return np.array(results)[:, ind].mean() if len(results) > 0 else 0
