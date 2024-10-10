import matplotlib.pyplot as plt
import numpy as np

from gibbs_ask import *
from likelihood_weighting import *
from rejection_sampling import *
from utils import *


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
