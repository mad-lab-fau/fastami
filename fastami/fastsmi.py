from typing import Sequence, Tuple, Union

import numpy as np
from numpy.random import BitGenerator, Generator, SeedSequence, default_rng
from scipy.stats import random_table
from sklearn.metrics.cluster import contingency_matrix, mutual_info_score


def standardized_mutual_info_mc(
    labels_true: Sequence[int],
    labels_pred: Sequence[int],
    seed: Union[None, SeedSequence, BitGenerator, Generator] = None,
    precision_goal: float = 0.1,
    min_samples: int = 1_000,
) -> Tuple[float, float]:
    """Calculate the Monte-Carlo estimate of the standardized mutual
    information sampling full contingency matrices directly.

    The Standardized Mutual Information (SMI) is defined as

        SMI(U, V) = (MI(U,V) - E(MI(U, V))) / std(MI(U, V)),

    under random permutations of the labels. The SMI can be interpreted as the
    number of standard deviations the observed mutual information is away from
    the expected value. The SMI should be used when comparing multiple clusterings
    via a common ground truth reference. The SMI is approximated via Monte-Carlo
    sampling of full contingency matrices with the corresponding marginals.

    For further information see Klede et al. (2023). FastAMI — A Monte Carlo
    Approach to the Adjustment for Chance in Clustering Comparison Metrics.
    Proceedings of the AAAI Conference on Artificial Intelligence.

    Args:
        labels_true: A clustering of the data into disjoint subsets.
        labels_pred: Another clustering of the data into disjoint subsets.
        seed: The random seed to use.
        precision_goal: Targeted relative or absolute error of the approximation.
            If the relative or the absolute error is smaller than this value, the algorithm stops.
        min_samples: Minimum number of samples to use.

    Returns:
        A tuple of the standardized mutual information and an error estimate.

    Notes:
        The error estimate assumes large sample sizes and can therefore be inaccurate if min_samples is too low.
    """
    prng = default_rng(seed=seed)
    contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    # Calculate the MI for the two clusterings
    mi = mutual_info_score(labels_true, labels_pred, contingency=contingency)
    # Calculate the expected value for the mutual information
    nrowt = np.ravel(contingency.sum(axis=1))
    ncolt = np.ravel(contingency.sum(axis=0))

    mc_samples = min_samples
    precision = precision_goal + 1
    smi_err = precision_goal + 1

    mi_arr = []

    while (precision > precision_goal) and (smi_err > precision_goal):
        for _ in range(mc_samples):
            mi_arr.append(mutual_info_score(_, _, contingency=random_table.rvs(nrowt, ncolt, random_state=prng)))

        emi = np.mean(mi_arr)
        emi_std = np.std(mi_arr, ddof=1)

        # Analytical continuation for zero variance
        if emi_std == 0.0:
            # If there is no variance, the expected value is the observed value
            # such that the analytical continuation is 1.0
            smi = 1.0
        else:
            smi = (mi - emi) / emi_std

        smi_err = np.sqrt(1 / len(mi_arr) + (smi * smi) / (2 * (len(mi_arr) - 1)))
        precision = abs(smi_err / smi)

        # Estimate samples needed to fulfill precision requirements

        s2 = smi**2
        ps2 = precision_goal**2 * s2

        mc_samples = np.ceil((2 + s2 + 2 * ps2 + np.sqrt(-16 * ps2 + (2 + s2 + 2 * ps2) ** 2)) / (4 * ps2)) - len(
            mi_arr
        )
        if mc_samples == np.nan:
            mc_samples = min_samples
        mc_samples = max(mc_samples, min_samples)
        # Make sure that we don't overestimate too much
        mc_samples = min(mc_samples, 100 * min_samples)
        mc_samples = int(mc_samples)

    return smi, smi_err
