import numpy as np

from utils import *


def likelihood_weighting(query, evidence, totalSamples=10000):
    """Perform likelihood weighting to estimate the normalized probability distribution of the query.

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
    totalWeights = 0
    for _ in range(totalSamples):
        sample = {}
        weight = 1
        for k in topological():
            node = Node.nodes[k]
            parent_list = tuple(sample[p] for p in node.parents)
            if k in evidence:
                sample[k] = evidence[k]
                prob = CPT(node, parent_list)
                weight *= prob if evidence[k] else (1 - prob)
            else:
                sample[k] = node.sample(parent_list)
        sample = np.array(list(sample.values()), dtype=float) * weight
        results.append(sample)
        totalWeights += weight
    return np.array(results)[:, ind].sum() / totalWeights if len(results) > 0 else 0
