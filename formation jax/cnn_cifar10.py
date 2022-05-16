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


class VGGBlock(hk.Module):
    def __init__(self, units, kernel_size=3, dropout=0.0):
        super().__init__()
        self.units = units
        self.kernel_size = kernel_size
        self.dropout = dropout

    def __call__(self, x, is_training):
        for _ in range(2):
            # Vscaling(2.0) -> He initialization
            x = hk.Conv2D(self.units, self.kernel_size,
                          w_init=Vscaling(2.0))(x)
            x = jax.nn.relu(x)
            # Batch norm: True, True -> create scale + offset params
            x = hk.BatchNorm(True, True, 0.99)(x, is_training)
        x = hk.max_pool(x, 2, 2, "SAME")
        if is_training and self.dropout > 0.0:
            x = hk.dropout(hk.next_rng_key(), self.dropout, x)
        return x


class VGG8:
    def __init__(self, blocks_units, blocks_dropout, final_dim, final_dropout,
                 n_classes):
        self.blocks_units = blocks_units
        self.blocks_dropout = blocks_dropout
        self.final_dim = final_dim
        self.final_dropout = final_dropout
        self.n_classes = n_classes

    @hk.transform_with_state
    def forward(self, x, is_training):
        # Convolutive blocks
        for i in range(len(self.blocks_units)):
            x = VGGBlock(
                self.blocks_units[i], dropout=self.blocks_dropout[i]
                )(x, is_training)
        # Linear part
        x = hk.Flatten()(x)
        x = hk.Linear(self.final_dim, w_init=Vscaling(2.0))(x)
        x = jax.nn.relu(x)
        x = hk.BatchNorm(True, True, 0.99)(x, is_training)
        if is_training:
            x = hk.dropout(hk.next_rng_key(), self.final_dropout, x)
        x = hk.Linear(self.n_classes, w_init=Vscaling(1.0, mode='fan_avg'))(x)
        return x  # logits

    def init(self, rng, x):
        return self.forward.init(rng, self, x, is_training=True)

    def apply(self, params, state, rng, x, is_training):
        return self.forward.apply(params, state, rng, self, x, is_training)

    def __call__(self, params, state, rng, x, is_training):
        return self.apply(params, state, rng, x, is_training)


def get_optimizer(**config):
    method = config.get('method', 'sgd')
    lr = config.get('lr', 0.01)
    weight_decay = config.get('weight_decay', 0.0)
    if method == 'sgd':
        optimizer = optax.chain(
            optax.add_decayed_weights(weight_decay),
            optax.scale(-lr)
            )
    elif method == 'adam':
        optimizer = optax.chain(
            optax.scale_by_adam(),
            optax.add_decayed_weights(weight_decay),
            optax.scale(-lr)
            )
    else:
        raise ValueError(f'Unknown optimizer method: {method} '
                         '(should be one of "sgd", "adam").')
    return optimizer


@partial(jit, static_argnums=[0, 6])
def evaluate(model, params, state, key, X, y, is_training):
    ypred, new_state = model(params, state, key, X, is_training)
    loss = optax.sigmoid_binary_cross_entropy(ypred, y).mean()
    metric = (jnp.argmax(ypred, axis=1) == jnp.argmax(y, axis=1)).mean()
    return loss, (new_state, metric)


@partial(jit, static_argnums=[0, 6])
def update(model, params, state, key, X, y, optimizer, opt_state):
    (loss, (new_state, metric)), grads = vgrad(evaluate, has_aux=True, argnums=1)(
        model, params, state, key, X, y, is_training=True
        )
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, new_state, new_opt_state, loss, metric


def fit(model, optimizer, data, key, **config):
    n_epochs = config['n_epochs']
    verbose = config['verbose']
    wandb = config['wandb']
    if wandb:
        import wandb
        wandb.init(project='jax-cnn-cifar10', config=config)

    # Initialization
    for X, _ in data['train'].take(1):  # take only first batch
        X = jnp.array(X)
        params, state = model.init(key, X)
        opt_state = optimizer.init(params)
    if verbose:
        print('Initialization done')

    history = {'train_loss': [], 'train_metric': [], 'val_loss': [],
               'val_metric': []}
    for epoch in range(1, n_epochs + 1):
        print(f'Epoch {epoch}/{n_epochs}')
        # Training loop
        train_losses, train_metrics = [], []
        for i_batch, (X_batch, y_batch) in enumerate(data['train']):
            X_batch, y_batch = jnp.array(X_batch), jnp.array(y_batch)
            params, state, opt_state, loss, metric = update(
                model, params, state, key, X_batch, y_batch, optimizer, opt_state
            )
            train_losses.append(np.array(loss))
            train_metrics.append(np.array(metric))
            if verbose:
                print(f'  batch {i_batch + 1}/{len(data["train"])}', end='\r')

        # Validation Loop
        val_losses, val_metrics = [], []
        for X_batch, y_batch in data['val']:
            X_batch, y_batch = jnp.array(X_batch), jnp.array(y_batch)
            loss, (state, metric) = evaluate(
                model, params, state, key, X_batch, y_batch, is_training=False
            )
            val_losses.append(np.array(loss))
            val_metrics.append(np.array(metric))

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


def test_loop(model, params, state, key, test_ds):
    test_losses, test_metrics = [], []
    for X_batch, y_batch in test_ds:
        X_batch, y_batch = jnp.array(X_batch), jnp.array(y_batch)
        loss, (state, metric) = evaluate(
            model, params, state, key, X_batch, y_batch, is_training=False
        )
        test_losses.append(np.array(loss))
        test_metrics.append(np.array(metric))
    return np.mean(test_losses), np.mean(test_metrics)


def run():
    seed = config['seed']

    # Get data, model and optimizer
    train_ds, val_ds, test_ds = load_cifar10(seed=seed, **config['data'])
    data = {'train': train_ds, 'val': val_ds, 'test': test_ds}
    model = VGG8(**config['model'])
    optimizer = get_optimizer(**config['optimizer'])

    # Fit model
    key = random.PRNGKey(seed)
    params, state, opt_state, history = fit(
        model, optimizer, data, key, **config['training']
        )

    # Save model
    if config['save_name']:  # not None and not empty string
        model_path = os.path.join('./models', config['save_name'])
        save_jax_model(model_path, params=params, state=state,
                       opt_state=opt_state, config_model=config['model'],
                       config_opt=config['optimizer'])

    # Plot history
    if config['plot']:
        plt_curves(history)

    # Test loop
    if config['test']:
        evals = test_loop(model, params, state, key, test_ds)
        plt_curves_test(*evals)

    if config['plot']:
        plt.show()


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
