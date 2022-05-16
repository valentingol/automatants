from functools import partial
import os

import haiku as hk
from haiku.initializers import VarianceScaling as Vscaling
import jax
from jax import jit, numpy as jnp, random, value_and_grad as vgrad
import matplotlib.pyplot as plt
import numpy as np
import optax

from utils.cifar10 import load_cifar10
from utils.plot import plt_curves, plt_curves_test
from utils.save_and_load import save_jax_model


def fit(model, optimizer, data, key, **config):
    # Initialization
    # TODO

    history = {'train_loss': [], 'train_metric': [], 'val_loss': [],
               'val_metric': []}
    for epoch in range(1, n_epochs + 1):
        print(f'Epoch {epoch}/{n_epochs}')
        # Training loop
        train_losses, train_metrics = [], []
        # TODO

        # Validation Loop
        val_losses, val_metrics = [], []
        # TODO

        evals = {
            'train_loss': np.mean(train_losses),
            'train_metric': np.mean(train_metrics),
            'val_loss': np.mean(val_losses),
            'val_metric': np.mean(val_metrics),
            }
        for k, v in evals.items():
            history[k].append(v)

        if verbose:
            for i, (k, v) in enumerate(evals.items()):
                if i in {0, 2}:
                    if i == 2: print()
                    print('  ', end='')
                print(f'{k}: {v:.4f} ', end='')
            print('\n')

        if wandb:
            wandb.log(evals)

    return params, state, opt_state, history


if __name__ == '__main__':
    # Configs
    config = {
        'seed': 0,
        'test': False,
        'plot': True,
        'save_name': 'my_vgg8',
        'data': {
            'batch_size': 128,
            'val_prop': 0.2,
            'test_prop': 0.1,
            },
        'training': {
            'wandb': False,
            'verbose': True,
            'n_epochs': 30,
            },
        'optimizer': {
            'method': 'adam',
            'lr': 0.01,
            'weight_decay': 0.03,
            },
        'model': {
            'blocks_units': [32, 64, 128],
            'blocks_dropout': [0.2, 0.3, 0.4],
            'final_dim': 128,
            'final_dropout': 0.5,
            'n_classes': 10,
            },
        }
    run()
