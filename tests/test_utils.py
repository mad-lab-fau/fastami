import numpy as np
from numpy.random import PCG64, Generator

from fastami.utils import WalkerRandomSampling


class TestUtils:
    def test_default_keys(self):
        prng = Generator(PCG64(12345))
        walker = WalkerRandomSampling([1, 2, 3], seed=prng)
        variates = [walker.random() for _ in range(10_000)]
        values, counts = np.unique(variates, return_counts=True)
        freqs = counts / len(variates)
        assert list(values) == [0, 1, 2]
        assert np.allclose(freqs, [1 / 6, 2 / 6, 3 / 6], atol=1e-2)

    def test_custom_keys(self):
        prng = Generator(PCG64(12345))
        walker = WalkerRandomSampling([1, 2, 3], ["a", "b", "c"], seed=prng)
        variates = [walker.random() for _ in range(10_000)]
        values, counts = np.unique(variates, return_counts=True)
        freqs = counts / len(variates)
        assert list(values) == ["a", "b", "c"]
        assert np.allclose(freqs, [1 / 6, 2 / 6, 3 / 6], atol=1e-2)

    def test_generator_weights(self):
        prng = Generator(PCG64(12345))

        class Weights:
            def __init__(self):
                self._weights = [1, 2, 3]

            def __iter__(self):
                return iter(self._weights)

            def __len__(self):
                return len(self._weights)

        weights = Weights()

        walker = WalkerRandomSampling(weights, seed=prng)
        variates = [walker.random() for _ in range(10_000)]
        values, counts = np.unique(variates, return_counts=True)
        freqs = counts / len(variates)
        assert list(values) == [0, 1, 2]
        assert np.allclose(freqs, [1 / 6, 2 / 6, 3 / 6], atol=1e-2)
