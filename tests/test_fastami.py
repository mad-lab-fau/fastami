import pytest
from sklearn.metrics.cluster import adjusted_mutual_info_score
from fastami import adjusted_mutual_info_mc
from numpy.random import PCG64, Generator
import numpy as np


class TestFastami:
    def test_invalid_input(self):
        with pytest.raises(ValueError):
            adjusted_mutual_info_mc([1, 2, 3], [1, 2])

    def test_empty_input(self):
        with pytest.raises(ValueError):
            adjusted_mutual_info_mc([], [])

    def test_single_label(self):
        labels = [0]
        ami, ami_err = adjusted_mutual_info_mc(labels, labels, seed=12345)
        assert abs(ami - adjusted_mutual_info_score(labels, labels)) <= ami_err

    def test_same_labels(self):
        labels = [-2, 0, -2, -2, 0, 1]
        ami, ami_err = adjusted_mutual_info_mc(labels, labels, seed=12345)
        assert abs(ami - 1) <= ami_err

        labels = [1, 2, 3, 4, 5, 6]
        ami, ami_err = adjusted_mutual_info_mc(labels, labels, seed=12345)
        assert abs(ami - 1) <= ami_err

    def test_different_labels(self):
        labels_true = [1, 1, 2, 2, 3, 3]
        labels_pred = [1, 2, 3, 1, 2, 3]
        ami, ami_err = adjusted_mutual_info_mc(labels_true, labels_pred, seed=12345)
        assert abs(ami - adjusted_mutual_info_score(labels_true, labels_pred)) <= ami_err

    def test_random_labels(self):
        prng = Generator(PCG64(12345))
        for _ in range(100):
            labels_true = prng.integers(0, 20, size=100)
            labels_pred = prng.integers(0, 20, size=100)
            ami, ami_err = adjusted_mutual_info_mc(
                labels_true, labels_pred, seed=prng, accuracy_goal=0.001, min_samples=1_000
            )
            assert abs(ami - adjusted_mutual_info_score(labels_true, labels_pred)) <= 0.01

    def test_accuracy(self):
        prng = Generator(PCG64(12345))
        labels_true = prng.integers(0, 10, size=100)
        labels_pred = prng.integers(0, 12, size=100)
        for accuracy_goal in [0.1, 0.01]:
            ami_sklearn = adjusted_mutual_info_score(labels_true, labels_pred)
            good_approximation = []
            for _ in range(100):
                ami, ami_err = adjusted_mutual_info_mc(
                    labels_true, labels_pred, seed=prng, accuracy_goal=0.01, min_samples=100
                )
                assert ami_err <= accuracy_goal
                good_approximation.append(abs(ami - ami_sklearn) <= ami_err)
            assert np.mean(good_approximation) >= 0.68
