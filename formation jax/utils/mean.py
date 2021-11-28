from jax import numpy as jnp

class Mean(object):
    def __init__(self):
        self.val = 0.0
        self.count = 0

    def __call__(self, val):
        if isinstance(val, jnp.ndarray):
            val = val.item()
        if val is not None:
            self.val = (self.val * self.count + val) / (self.count + 1)
            self.count += 1
        return self.val

    def reset(self):
        self.val = 0.0
        self.count = 0

    def __repr__(self, *args, **kwargs):
        return self.val.__repr__(*args, **kwargs)

    def __str__(self, *args, **kwargs):
        return self.val.__str__(*args, **kwargs)

    def __format__(self, *args, **kwargs) -> str:
        return self.val.__format__(*args, **kwargs)


class MultiMean(object):
    def __init__(self, n_means=1):
        self.n_means = n_means
        self.means = [Mean() for _ in range(n_means)]

    def __call__(self, *vals):
        return tuple(self.means[i](vals[i]) for i in range(self.n_means))

    def reset(self, indicies=None):
        if isinstance(indicies, int):
            indicies = [indicies]
        if indicies is None:
            indicies = list(range(self.n_means))
        for i in indicies:
            self.means[i].reset()

    @property
    def values(self):
        return tuple(self.means[i].val for i in range(self.n_means))

    def __repr__(self, *args, **kwargs):
        return self.means.__repr__(*args, **kwargs)

    def __str__(self, *args, **kwargs):
        return self.means.__str__(*args, **kwargs)
