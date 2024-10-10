import json
import random

import matplotlib.pyplot as plt
import numpy as np


class TupleError(Exception):
    """Raise when tuple is not present in conditional probability dictionary.
    This is here mainly for students to catch their own errors.
    """

    def __init__(self, name):
        super().__init__(
            "Error: Tuple not present in conditional probability dictionary of Node " + name
        )


class Node:
    """A node in the bayesian network.
    Please read all comments in this class.
    """

    nodes = {}  # Static object: use this to access any existing node

    # REMEMBER TO RESET THIS if you would like to use a different JSON file
    def __init__(self, name):
        self.name = name  # string
        self.parents = []  # list of parent names
        self.children = []  # list of children names
        self.condProbs = {}  # (boolean tuple in parent names order) --> float
        self.prob = -1  # If no parent, use this
        Node.nodes[name] = self

    def sample(self, tup):
        """Sample based on boolean tuple passed in.
        Return bool of conditional event

        Arguments:
        tup -- boolean tuple, in order of self.parents
        """
        true_prob = self.prob
        # Usually not the root node in bayesian network
        if self.condProbs:
            if tup not in self.condProbs:
                raise TupleError(self.name)
            true_prob = self.condProbs[tup]
        return True if random.random() < true_prob else False


def topological():
    """Return list of Node names in topological order.
    The code here performs a topological sort on the nodes.
    """
    visited = set()
    top_sorted = []

    # Helper function to DFS over all children
    def visit(node_name):
        visited.add(node_name)
        for child_name in Node.nodes[node_name].children:
            if child_name not in visited:
                visit(child_name)
        # At this point, we have visited all children
        top_sorted.append(node_name)

    for node_name in Node.nodes:
        if node_name not in visited:
            visit(node_name)
    return top_sorted[::-1]


def parse_file(filename):
    """Parse JSON file of bayesian network.

    JSON key-value pairs:
    "Name"         -- Name of the node.
    "Parents"      -- List of names of parent nodes.
                      Conditionals are given in order of this list.
    "Children"     -- List of names of child nodes.
    "Conditionals" -- Single float OR List of conditional probabilities.
                      **float for a root node, list of floats for a non-root node**
                      Indices are integer representation of bits (i.e. 0=FF, 1=FT, 2=TF, 3=TT).
                      Ordering of parents align with the bits here
                          i.e. parents = ['Sprinkler', 'Rain']
                               FT refers to Sprinkler=False, Rain=True
    """
    with open(filename, "r") as f:
        data = json.load(f)
        for node_data in data:
            node = Node(node_data["Name"])
            node.parents = node_data["Parents"]
            node.children = node_data["Children"]

            # A root node in bayesian network
            if type(node_data["Conditionals"]) is not list:
                node.prob = node_data["Conditionals"]

            # A non-root node in bayesian network
            else:
                # Parse bit representation of each index in the list
                for bits, prob in enumerate(node_data["Conditionals"]):
                    tup = []
                    for event in reversed(node.parents):
                        tup.append(bool(bits & 1))
                        bits = bits >> 1
                    node.condProbs[tuple(reversed(tup))] = prob


def CPT(node, parent_list):
    """Return the conditional probability of the node given its parents.

    Parameters:
    -----------
    node : Node
        The node whose conditional probability needs to be returned.
    parent_list : tuple of bool
        The values of the node's parents in a tuple.

    Returns:
    --------
    float : The conditional probability for the node.
    """
    if not node.parents:
        return node.prob
    else:
        return node.condProbs[parent_list]


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


def run_experiment_with_seed(algorithm, query, evidence, sample_size, true_prob, repetitions=100):
    """
    Run the experiment with different random seeds for a given sampling algorithm.

    Parameters:
    -----------
    algorithm : function
        The sampling algorithm to run (e.g., rejection_sampling, likelihood_weighting, gibbs_ask).
    query : str
        The name of the query node.
    evidence : dict
        A dictionary mapping node names (strings) to boolean values.
    sample_size : int
        The number of samples to generate in each experiment.
    true_prob : float
        The true probability to compare against for error calculation.
    repetitions : int, optional
        The number of repetitions with different random seeds (default is 100).

    Returns:
    --------
    tuple of float : mean_error, std_dev
        The mean error and standard deviation of the errors across the repetitions.
    """
    errors = []
    for i in range(repetitions):
        random.seed(i)  # Set a different seed for each repetition
        result = algorithm(query, evidence, sample_size)
        error = abs(result - true_prob)
        errors.append(error)
    mean_error = np.mean(errors)
    std_dev = np.std(errors)
    return mean_error, std_dev


def plot_all_algorithms(
    rejection_means, rejection_stds, likelihood_means, likelihood_stds, gibbs_means, gibbs_stds
):
    """
    Plot the error vs sample size with standard deviations for three algorithms.

    Parameters:
    -----------
    rejection_means : list of float
        Mean errors for rejection sampling at different sample sizes.
    rejection_stds : list of float
        Standard deviations for rejection sampling at different sample sizes.
    likelihood_means : list of float
        Mean errors for likelihood weighting at different sample sizes.
    likelihood_stds : list of float
        Standard deviations for likelihood weighting at different sample sizes.
    gibbs_means : list of float
        Mean errors for Gibbs sampling at different sample sizes.
    gibbs_stds : list of float
        Standard deviations for Gibbs sampling at different sample sizes.

    Returns:
    --------
    None
    """
    plt.figure()
    # Plot the error curve for rejection sampling, with error bars representing standard deviation
    plt.errorbar(
        sample_sizes, rejection_means, yerr=rejection_stds, label="Rejection Sampling", fmt="-o"
    )
    # Plot the error curve for likelihood weighting, with error bars representing standard deviation
    plt.errorbar(
        sample_sizes,
        likelihood_means,
        yerr=likelihood_stds,
        label="Likelihood Weighting",
        fmt="-s",
    )
    # Plot the error curve for Gibbs sampling, with error bars representing standard deviation
    plt.errorbar(sample_sizes, gibbs_means, yerr=gibbs_stds, label="Gibbs Sampling", fmt="-^")

    plt.xlabel("Number of Samples")
    plt.ylabel("Mean Error")
    plt.title("Mean Error vs Number of Samples (with Standard Deviation)")
    plt.legend()
    plt.savefig("./error_vs_samples.png", dpi=300)
    plt.show()


def main(query, evidence, sample_sizes, true_prob, repetitions):
    """
    Run the sampling algorithms and plot the results for different sample sizes.

    Parameters:
    -----------
    query : str
        The name of the query node.
    evidence : dict
        A dictionary mapping node names (strings) to boolean values.
    sample_sizes : list of int
        List of sample sizes to iterate through.
    true_prob : float
        The true probability for comparison.
    repetitions : int
        The number of repetitions with different random seeds.

    Returns:
    --------
    None
    """
    rejection_means, rejection_stds = [], []
    likelihood_means, likelihood_stds = [], []
    gibbs_means, gibbs_stds = [], []
    # Iterate through the defined sample sizes
    for sample_size in sample_sizes:
        # Rejection Sampling
        mean_error, std_dev = run_experiment_with_seed(
            rejection_sampling, query, evidence, sample_size, true_prob, repetitions
        )
        rejection_means.append(mean_error)
        rejection_stds.append(std_dev)
        # Likelihood Weighting
        mean_error, std_dev = run_experiment_with_seed(
            likelihood_weighting, query, evidence, sample_size, true_prob, repetitions
        )
        likelihood_means.append(mean_error)
        likelihood_stds.append(std_dev)
        # Gibbs Sampling
        mean_error, std_dev = run_experiment_with_seed(
            gibbs_ask, query, evidence, sample_size, true_prob, repetitions
        )
        gibbs_means.append(mean_error)
        gibbs_stds.append(std_dev)
    # Plot the results for all three algorithms
    plot_all_algorithms(
        rejection_means, rejection_stds, likelihood_means, likelihood_stds, gibbs_means, gibbs_stds
    )


if __name__ == "__main__":
    parse_file("./weather.json")
    evidence = {"Cloudy": False, "WetGrass": True}
    query = "Rain"

    true_prob = 21 / 61  # 0.3442622950819672
    # First, test the three algorithms with sample size of 10000
    random.seed(42)
    print("\nrejection_sampling:")
    print(rejection_sampling(query, evidence, 10000))
    print("\nlikelihood_weighting:")
    print(likelihood_weighting(query, evidence, 10000))
    print("\ngibbs_ask:")
    print(gibbs_ask(query, evidence, 10000))

    # Now, run the experiment with different sample sizes and seeds, then plot the results
    sample_sizes = list(range(10, 101, 10))
    repetitions = 100  # with random seeds 0, 1, 2, ..., 99
    main(query, evidence, sample_sizes, true_prob, repetitions)
