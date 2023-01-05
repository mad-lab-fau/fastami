from typing import Sequence, Tuple, Union
from numpy import ravel, unique, log, sqrt, ceil, mean, std, finfo
from numpy.random import default_rng, SeedSequence, BitGenerator, Generator
from sklearn.metrics.cluster import contingency_matrix, mutual_info_score, entropy
from fastami.utils import WalkerRandomSampling


def adjusted_mutual_info_mc(
    labels_true: Sequence[int],
    labels_pred: Sequence[int],
    seed: Union[None, SeedSequence, BitGenerator, Generator] = None,
    accuracy_goal: float = 0.01,
    min_samples: int = 10_000,
) -> Tuple[float, float]:
    """Approximate adjusted mutual information score for two clusterings.

    The ajusted mutual information score is calculated based on a Monte-Carlo
    estimate of the expected mutual information.

    Args:
        labels_true: A clustering of the data into disjoint subsets.
        labels_pred: Another clustering of the data into disjoint subsets.
        seed: The random seed to use.
        accuracy_goal: The desired accuracy of the Monte-Carlo estimate.
        min_samples: The minimum number of samples to use.

    Returns:
       A tuple of the adjusted mutual information and an error estimate.

    Raises:
        ValueError: If the length of the labels do not match or if the labels
            are empty.
    """
    if len(labels_true) != len(labels_pred):
        raise ValueError("labels_true and labels_pred must have the same length.")
    if len(labels_true) < 1:
        raise ValueError("labels_true and labels_pred must not be empty.")

    n_total = len(labels_true)

    contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    # Calculate the MI for the two clusterings
    mi = mutual_info_score(labels_true, labels_pred, contingency=contingency)
    # Calculate the expected value for the mutual information
    nrowt = ravel(contingency.sum(axis=1))
    ncolt = ravel(contingency.sum(axis=0))

    if len(nrowt) == 1 and len(ncolt) == 1:
        # Special case where both clusterings have only one cluster
        return 1.0, 0.0

    # Calculate entropy for each labeling
    h_true, h_pred = entropy(labels_true), entropy(labels_pred)
    normalizer = 0.5 * (h_true + h_pred)

    # Expected mutual information
    # Due to normalization of nrowt_counts and ncolt_counts and 1/N**2 after rescaling hypergeometric pdf
    normalization = len(nrowt) / n_total * len(ncolt) / n_total
    prng = default_rng(seed=seed)

    nrowt_vals, nrowt_counts = unique(nrowt, return_counts=True)
    ncolt_vals, ncolt_counts = unique(ncolt, return_counts=True)

    nrowt_rng = WalkerRandomSampling(weights=nrowt_counts, keys=nrowt_vals, seed=prng)
    ncolt_rng = WalkerRandomSampling(weights=ncolt_counts, keys=ncolt_vals, seed=prng)

    mc_samples = min_samples
    ami_err = accuracy_goal + 1
    emi_arr = []

    while ami_err > accuracy_goal:
        a_arr = nrowt_rng.random(mc_samples)
        b_arr = ncolt_rng.random(mc_samples)
        n_arr = prng.hypergeometric(ngood=a_arr - 1, nbad=n_total - a_arr, nsample=b_arr - 1) + 1

        # Watch out for potential underflows.
        emi_arr.extend((a_arr * b_arr) * log(n_total * n_arr / (a_arr * b_arr)))
        emi = normalization * mean(emi_arr)

        denominator = normalizer - emi
        # Avoid 0.0 / 0.0 when expectation equals maximum, i.e a perfect match.
        # normalizer should always be >= emi, but because of floating-point
        # representation, sometimes emi is slightly larger. Correct this
        # by preserving the sign.
        if denominator < 0:
            denominator = min(denominator, -finfo("float64").eps)
        else:
            denominator = max(denominator, finfo("float64").eps)

        ami = (mi - emi) / denominator

        # Avoid division by zero.
        second_denominator = max(abs(mi - emi), finfo("float64").eps)

        ami_std = (
            normalization
            * std(emi_arr, ddof=1)
            * abs(ami)
            * abs(normalizer - mi)
            / (abs(denominator) * second_denominator)
        )
        ami_err = ami_std / sqrt(len(emi_arr))

        mc_samples = int(ceil((ami_std / accuracy_goal) ** 2 - len(emi_arr)))

        mc_samples = max(mc_samples, min_samples)
        # Make sure that we don't overestimate too much
        mc_samples = min(mc_samples, 100 * min_samples)

    return ami, ami_err
