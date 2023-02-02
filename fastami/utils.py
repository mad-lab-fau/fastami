from numpy import array, ndarray, ones, where
from numpy.random import default_rng


class WalkerRandomSampling(object):
    """Walker's alias method for random objects with different probablities.

    Based on the implementation of Denis Bzowy at the following URL:
    http://code.activestate.com/recipes/576564-walkers-alias-method-for-random-objects-with-diffe/
    licensed under the MIT license.
    """

    def __init__(self, weights, keys=None, seed=None):
        """Builds the Walker tables ``prob`` and ``inx`` for calls to `random()`.
        The weights (a list or tuple or iterable) can be in any order and they
        do not even have to sum to 1.

        Args:
            weights: Weights of the random variates.
            keys: Keys of the random variates.
            seed: Seed for the random number generator.

        Raises:
            ValueError: If the weights do not sum to 1.
        """
        n = self.n = len(weights)
        if keys is None:
            self.keys = keys
        else:
            self.keys = array(keys)

        self._rng = default_rng(seed)

        if isinstance(weights, (list, tuple)):
            weights = array(weights, dtype=float)
        elif isinstance(weights, ndarray):
            if weights.dtype != float:
                weights = weights.astype(float)
        else:
            weights = array(list(weights), dtype=float)

        weights = weights * n / weights.sum()

        inx = -ones(n, dtype=int)
        short = where(weights < 1)[0].tolist()
        long = where(weights > 1)[0].tolist()
        while short and long:
            j = short.pop()
            k = long[-1]

            inx[j] = k
            weights[k] -= 1 - weights[j]
            if weights[k] < 1:
                short.append(k)
                long.pop()

        self.prob = weights
        self.inx = inx

    def random(self, count=None):
        """Returns a given number of random integers or keys, with probabilities
        being proportional to the weights supplied in the constructor.
        When `count` is ``None``, returns a single integer or key, otherwise
        returns a NumPy array with a length given in `count`.

        Args:
            count: Number of random integers or keys to return.

        Returns:
            Random variates with probabilities being proportional to the weights supplied in the constructor.
        """
        if count is None:
            u = self._rng.random()
            j = self._rng.integers(self.n)
            k = j if u <= self.prob[j] else self.inx[j]
            return self.keys[k] if self.keys is not None else k

        u = self._rng.random(size=count)
        j = self._rng.integers(self.n, size=count)
        k = where(u <= self.prob[j], j, self.inx[j])
        return self.keys[k] if self.keys is not None else k
