from functools import partial
import os

import haiku as hk
from haiku.initializers import VarianceScaling as Vscaling
import jax
from jax import jit, numpy as jnp, random, value_and_grad as vgrad
import matplotlib.pyplot as plt
import optax

from utils.cifar10 import load_cifar10
from utils.mean import MultiMean
from utils.plot import plt_curves, plt_curves_test
from utils.save_and_load import save_jax_model, load_jax_model


if __name__ == '__main__':
    # Configs
    save_name = 'My_vgg8' # None or empty to not save
    seed = 0
    n_epochs = 30
    batch_size = 128
    val_prop = 0.08
    test_prop = 0.16
    lr = 0.01
    verbose = True
    test = False

    key = random.PRNGKey(seed)

    train_ds, val_ds, test_ds = load_cifar10(batch_size=batch_size,
                                             val_prop=val_prop,
                                             test_prop=test_prop,
                                             seed=seed)
    # Train + Validation
    # ...
    params, state = None


    if save_name is not None and save_name != '':
        # Save model
        model_path = os.path.join('./models', save_name)
        save_jax_model(params, state, model_path)
        # Load it to test the saved model
        params, state = load_jax_model(model_path)

    # Test loop
    # ...

    # Show the plots
    plt.show()
