import numpy as np

from utils import *


def Markov_blanket_prob(node_name, sample):
    """Calculate the conditional probability of the node given its Markov blanket on current sample.

    Parameters:
    -----------
    node_name : str
        The name of the node.
    sample : dict
        A dictionary mapping node names to boolean values (the current sample).

    Returns:
    --------
    float : The normalized probability of the node.
    """
    node = Node.nodes[node_name]
    parent_list = tuple(sample[p] for p in node.parents)
    Blanket_True = CPT(node, parent_list)
    Blanket_False = 1 - Blanket_True
    for child_name in node.children:
        child = Node.nodes[child_name]
        child_parent_True = tuple(sample[p] if p != node_name else True for p in child.parents)
        child_parent_False = tuple(sample[p] if p != node_name else False for p in child.parents)
        child_True = CPT(child, child_parent_True)
        child_False = CPT(child, child_parent_False)
        if sample[child_name]:
            Blanket_True *= child_True
            Blanket_False *= child_False
        else:
            Blanket_True *= 1 - child_True
            Blanket_False *= 1 - child_False
    # return normalized probability
    return Blanket_True / (Blanket_True + Blanket_False)


def gibbs_ask(query, evidence, totalSamples=10000):
    """Perform Gibbs sampling to estimate the normalized probability distribution of the query.

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

    sample = {k: random.choice([0, 1]) for k in topological()}
    for k in evidence:
        sample[k] = evidence[k]

    for _ in range(totalSamples):
        for k in topological():
            if k not in evidence:
                Blanket_True = Markov_blanket_prob(k, sample)
                sample[k] = random.random() < Blanket_True
        results.append(list(sample.values()))
    return np.array(results)[:, ind].mean() if len(results) > 0 else 0
