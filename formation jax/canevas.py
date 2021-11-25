from functools import partial
import os

import haiku as hk
import jax
import jax.dlpack
from jax import random, grad, jit
from jax import numpy as jnp
import matplotlib.pyplot as plt
import optax

from utils.cifar10 import load_cifar10
from utils.save_and_load import save_jax_model, load_jax_model
from utils.plot import plt_curves, plt_curves_test



if __name__ == '__main__':
    # Configs
    pass
