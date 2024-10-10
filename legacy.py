import random

import matplotlib.pyplot as plt
import numpy as np


class Node:
    def __init__(self):
        """Define a base class for nodes in the Bayesian network."""
        self.prob = None  # Probability

    def set_prob(self, prob):
        """Set the probability of the node.

        Parameters:
        -----------
        prob : float
            The probability value to set.

        Returns:
        --------
        Node : The node object with the updated probability.
        """
        self.prob = prob
        return self

    def sample(self):
        """
        Sample a value (True/False) based on the node's probability.
        Returns:
        --------
        bool : A boolean value representing the sampled result (True if sampled value is below prob).
        """
        return random.random() < self.prob


class C(Node):
    def __init__(self, tup):
        """Define the class for Cloudy node (C)"""
        super().__init__()
        # Cloudy has a fixed probability of 0.5
        self.prob = 0.5


class R(Node):
    def __init__(self, tup):
        """Define the class for Rain node (R) conditioned on Cloudy

        Parameters:
        -----------
        tup : list
            A tuple containing the sampled value for C (Cloudy).
        """
        super().__init__()
        # If Cloudy is true, Rain has a higher probability, otherwise it's lower
        self.prob = 0.8 if tup[0] else 0.2


class S(Node):
    def __init__(self, tup):
        """Define the class for Sprinkler node (S) conditioned on Cloudy

        Parameters:
        -----------
        tup : list
            A tuple containing the sampled value for C (Cloudy).
        """
        super().__init__()
        # If Cloudy is true, Sprinkler has a lower probability, otherwise it's higher
        self.prob = 0.1 if tup[0] else 0.5


class W(Node):
    def __init__(self, tup):
        """Define the class for WetGrass node (W) conditioned on Sprinkler and Rain

        Parameters:
        -----------
        tup : list
            A tuple containing the sampled values for Sprinkler and Rain.
        """
        super().__init__()
        # Conditional probabilities for WetGrass based on Sprinkler and Rain states
        if tup[1] and tup[2]:
            self.prob = 0.99  # Both Sprinkler and Rain true
        elif not tup[1] and not tup[2]:
            self.prob = 0  # Both Sprinkler and Rain false
        else:
            self.prob = 0.9  # One of them is true


def rejection_sampling(totalSamples=10000):
    """
    Perform rejection sampling to estimate the probability.

    Parameters:
    -----------
    totalSamples : int, optional
        The number of samples to draw (default is 10000).

    Returns:
    --------
    float : The estimated probability based on the samples.
    """
    results = []
    for _ in range(totalSamples):
        tup = []
        # Generate samples for each node
        for node in [C, R, S, W]:
            tup.append(node(tup).sample())
        # Consider only if Cloudy is false and WetGrass is true
        if not tup[0] and tup[-1]:
            results.append(tup)
    # Return the estimated probability of Rain
    return np.array(results)[:, 1].mean() if len(results) > 0 else 0


def likelihood_weighting(totalSamples=10000):
    """
    Perform likelihood weighting to estimate the probability.

    Parameters:
    -----------
    totalSamples : int, optional
        The number of samples to draw (default is 10000).

    Returns:
    --------
    float : The estimated probability based on the samples.
    """
    results = []
    totalWeights = 0
    for _ in range(totalSamples):
        tup = [False]  # Cloudy is false
        tup.append(R(tup).sample())  # Sample Rain
        tup.append(S(tup).sample())  # Sample Sprinkler
        tup.append(True)  # WetGrass is true
        weight = (1 - C(tup).prob) * W(tup).prob
        tup = np.array(tup, dtype=float) * weight  # Weight the sample
        results.append(tup)
        totalWeights += weight
    # Return the estimated probability of Rain
    return np.array(results)[:, 1].sum() / totalWeights if len(results) > 0 else 0


# def Markov_blanket_prob(evidence, ind, goal, totalSamples=100000):
#     """
#     Calculate the conditional probability of the node given its Markov blanket on current sample. Using rejection sampling to estimate the probability.

#     Parameters:
#     -----------
#     evidence : array-like
#         An array containing the evidence variables.
#     ind : list of int
#         Indices of the evidence variables in the sample.
#     goal : int
#         Index of the goal variable for which the probability is calculated.
#     totalSamples : int, optional
#         The number of samples to generate (default is 100000).

#     Returns:
#     --------
#     float : The estimated conditional probability.
#     """
#     results = []
#     for i in range(totalSamples):
#         tup = []
#         for _ in [C, R, S, W]:
#             tup.append(_(tup).sample())  # Generate samples for each node
#         tup = np.array(tup)
#         if np.array_equal(evidence[ind], tup[ind]):  # Compare evidence
#             results.append(tup)
#     # Return the probability for the goal variable
#     return np.array(results)[:, goal].mean() if len(results) > 0 else 0


# Calculate usful conditional probabilities using Markov blanket(Rejection sampling)
# P(R=T | C=F, S=T, W=T)
# evidence = np.array([0, 1, 1, 1])
# prob_s_T = Markov_blanket_prob(evidence, [0, 2, 3], 1)
# # P(R=T | C=F, S=F, W=T)
# evidence = np.array([0, 1, 0, 1])
# prob_s_F = Markov_blanket_prob(evidence, [0, 2, 3], 1)
# # P(S=T | C=F, R=T, W=T)
# evidence = np.array([0, 1, 1, 1])
# prob_r_T = Markov_blanket_prob(evidence, [0, 1, 3], 2)
# # P(S=T | C=F, R=F, W=T)
# evidence = np.array([0, 0, 1, 1])
# prob_r_F = Markov_blanket_prob(evidence, [0, 1, 3], 2)

# ground truth
# prob_s_T, prob_s_F, prob_r_T, prob_r_F = 0.2156862745098039, 1, 0.5238095238095237, 1
# print(prob_s_T, prob_s_F, prob_r_T, prob_r_F)


# def gibbs_ask(totalSamples=10000):
#     """
#     Perform Gibbs sampling to estimate the probability.

#     Parameters:
#     -----------
#     totalSamples : int, optional
#         The number of samples to generate (default is 10000).

#     Returns:
#     --------
#     float : The estimated probability based on the samples.
#     """
#     results = []
#     tup = [random.choice([0, 1]) for _ in range(4)]  # Initialize random states
#     tup[0], tup[-1] = False, True  # Fix Cloudy to false and WetGrass to true
#     for _ in range(totalSamples):
#         # Resample Rain
#         prob = prob_s_T if tup[2] else prob_s_F
#         tup[1] = random.random() < prob
#         # Resample Sprinkler
#         prob = prob_r_T if tup[1] else prob_r_F
#         tup[2] = random.random() < prob
#         results.append(np.array(tup))
#         # results.append(tup)  - incorrect
#         """Note: Gibbs sampling is different from Rejection sampling and Likelihood weighting. It starts from a random state and the algorithm converges to the true distribution as the number of samples increases. Which means the output 'tup' in current for iteration is the initial 'tup' of the next iteration. It's somehow like evolving the state of the system.
#         So when we wanna store 'tup' in the last step, we should not append 'tup' itself but a copy of 'tup' to 'results', otherwise next iteration will change the tup value that are already stored in 'results' list as they refer to the same Memory Address.
#         """
#     # Return the estimated probability of Rain
#     return np.array(results)[:, 1].mean() if len(results) > 0 else 0


def Markov_blanket_prob_new(node_ind, tup):
    """Calculate the conditional probability of the node given its Markov blanket on current sample.

    Parameters:
    -----------
    node_ind : int
        The index of the node.
    tup : list
        A list mapping node names to boolean values (the current sample).

    Returns:
    --------
    float : The normalized probability of the node.
    """

    node = [C, R, S, W][node_ind]
    prob_True = node(tup).prob
    prob_False = 1 - prob_True

    tup_T = tup.copy()
    tup_T[node_ind] = True
    tup_F = tup.copy()
    tup_F[node_ind] = False
    # for every child node, calculate the probability
    child_True = W(tup_T).prob
    child_False = W(tup_F).prob
    if tup[-1]:
        prob_True *= child_True
        prob_False *= child_False
    else:
        prob_True *= 1 - child_True
        prob_False *= 1 - child_False
    # return normalized probability
    return prob_True / (prob_True + prob_False)


def gibbs_ask(totalSamples=10000):
    """
    Perform Gibbs sampling to estimate the probability.

    Parameters:
    -----------
    totalSamples : int, optional
        The number of samples to generate (default is 10000).

    Returns:
    --------
    float : The estimated probability based on the samples.
    """
    results = []
    tup = [random.choice([0, 1]) for _ in range(4)]  # Initialize random states
    tup[0], tup[-1] = False, True  # Fix Cloudy to false and WetGrass to true
    for _ in range(totalSamples):
        for node in [R, S]:
            node_ind = [C, R, S, W].index(node)
            prob = Markov_blanket_prob_new(node_ind, tup)
            tup[node_ind] = node(tup).set_prob(prob).sample()
        results.append(np.array(tup))
        # results.append(tup)  - incorrect
        """Note: Gibbs sampling is different from Rejection sampling and Likelihood weighting. It starts from a random state and the algorithm converges to the true distribution as the number of samples increases. Which means the output 'tup' in current for iteration is the initial 'tup' of the next iteration. It's somehow like evolving the state of the system.
        So when we wanna store 'tup' in the last step, we should not append 'tup' itself but a copy of 'tup' to 'results', otherwise next iteration will change the tup value that are already stored in 'results' list as they refer to the same Memory Address.
        """
    # Return the estimated probability of Rain
    return np.array(results)[:, 1].mean() if len(results) > 0 else 0


def run_experiment_with_seed(algorithm, sample_size, true_prob, repetitions=100):
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
        result = algorithm(sample_size)
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


def main(sample_sizes, true_prob, repetitions):
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
            rejection_sampling, sample_size, true_prob, repetitions
        )
        rejection_means.append(mean_error)
        rejection_stds.append(std_dev)
        # Likelihood Weighting
        mean_error, std_dev = run_experiment_with_seed(
            likelihood_weighting, sample_size, true_prob, repetitions
        )
        likelihood_means.append(mean_error)
        likelihood_stds.append(std_dev)
        # Gibbs Sampling
        mean_error, std_dev = run_experiment_with_seed(
            gibbs_ask, sample_size, true_prob, repetitions
        )
        gibbs_means.append(mean_error)
        gibbs_stds.append(std_dev)
    # Plot the results for all three algorithms
    plot_all_algorithms(
        rejection_means, rejection_stds, likelihood_means, likelihood_stds, gibbs_means, gibbs_stds
    )


if __name__ == "__main__":
    true_prob = 21 / 61  # 0.3442622950819672
    # First, test the three algorithms with sample size of 10000
    random.seed(42)
    print("\nrejection_sampling:")
    print(rejection_sampling(10000))
    print("\nlikelihood_weighting:")
    print(likelihood_weighting(10000))
    print("\ngibbs_ask:")
    print(gibbs_ask(10000))

    # Now, run the experiment with different sample sizes and seeds, then plot the results
    sample_sizes = list(range(10, 101, 10))
    repetitions = 100  # with random seeds 0, 1, 2, ..., 99
    main(sample_sizes, true_prob, repetitions)
