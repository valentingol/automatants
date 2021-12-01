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

class VGG8(hk.Module):
    def __init__(self):
        super().__init__()
        # Block 1
        self.conv1_1 = hk.Conv2D(32, 3, w_init=Vscaling(2.0))
        self.bn1_1 = hk.BatchNorm(True, True, 0.99)
        self.conv1_2 = hk.Conv2D(32, 3, w_init=Vscaling(2.0))
        self.bn1_2 = hk.BatchNorm(True, True, 0.99)
        # Block 2
        self.conv2_1 = hk.Conv2D(64, 3, w_init=Vscaling(2.0))
        self.bn2_1 = hk.BatchNorm(True, True, 0.99)
        self.conv2_2 = hk.Conv2D(64, 3, w_init=Vscaling(2.0))
        self.bn2_2 = hk.BatchNorm(True, True, 0.99)
        # Block 2
        self.conv3_1 = hk.Conv2D(128, 3, w_init=Vscaling(2.0))
        self.bn3_1 = hk.BatchNorm(True, True, 0.99)
        self.conv3_2 = hk.Conv2D(128, 3, w_init=Vscaling(2.0))
        self.bn3_2 = hk.BatchNorm(True, True, 0.99)
        # Linear part
        self.lin1 = hk.Linear(128, w_init=Vscaling(2.0))
        self.bn4 = hk.BatchNorm(True, True, 0.99)
        self.lin2 = hk.Linear(10, w_init=Vscaling(1.0, "fan_avg"))

    def forward(self, x, is_training):
        # Block 1
        x = jax.nn.relu(self.conv1_1(x))
        x = self.bn1_1(x, is_training)
        x = jax.nn.relu(self.conv1_2(x))
        x = self.bn1_2(x, is_training)
        x = hk.max_pool(x, 2, 2, "SAME")
        if is_training:
            x = hk.dropout(hk.next_rng_key(), 0.2, x)
        # Block 2
        x = jax.nn.relu(self.conv2_1(x))
        x = self.bn2_1(x, is_training)
        x = jax.nn.relu(self.conv2_2(x))
        x = self.bn2_2(x, is_training)
        x = hk.max_pool(x, 2, 2, "SAME")
        if is_training:
            x = hk.dropout(hk.next_rng_key(), 0.3, x)
        # Block 3
        x = jax.nn.relu(self.conv3_1(x))
        x = self.bn3_1(x, is_training)
        x = jax.nn.relu(self.conv3_2(x))
        x = self.bn3_2(x, is_training)
        x = hk.max_pool(x, 2, 2, "SAME")
        if is_training:
            x = hk.dropout(hk.next_rng_key(), 0.4, x)
        # Linear part
        x = hk.Flatten()(x)
        x = jax.nn.relu(self.lin1(x))
        x = self.bn4(x, is_training)
        if is_training:
            x = hk.dropout(hk.next_rng_key(), 0.5, x)
        x = self.lin2(x)
        return x # logits

    def __call__(self, x, is_training):
        return self.forward(x, is_training)


@hk.transform_with_state
def forward(X, is_training=True):
    vgg8 = VGG8()
    return vgg8(X, is_training)


def init(key, X, lr):
    params, state = forward.init(key, X, True)
    optimizer = optax.chain(
        optax.scale_by_adam(),
        optax.add_decayed_weights(0.03),
        optax.scale(-lr)
    )
    opt_state = optimizer.init(params)
    return params, state, opt_state, optimizer


@partial(jit, static_argnums=5)
def evaluate(params, state, key, X, y, is_training):
    ypred, new_state = forward.apply(params, state, key, X, is_training)
    loss = optax.sigmoid_binary_cross_entropy(ypred, y).mean()
    metric = (jnp.argmax(ypred, axis=1) == jnp.argmax(y, axis=1)).mean()
    return loss, (new_state, metric)


@partial(jit, static_argnums=5)
def update(params, state, key, X, y, optimizer, opt_state):
    (loss, (new_state, metric)), grads = vgrad(evaluate, has_aux=True)(
        params, state, key, X, y, is_training=True
        )
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, new_state, new_opt_state, loss, metric


def train_validation_loop(key, train_ds, val_ds, n_epochs, lr, verbose=True):
    # Initialization
    for X, _ in train_ds.take(1): # go only once to get the shape
        X = jnp.array(X)
        params, state, opt_state, optimizer = init(key, X, lr)
    if verbose:
        print('Initialization done')

    # evals: train_loss, train_metric, val_loss, val_metric
    evals = MultiMean(4)
    evals_epoch = [[] for _ in range(4)]
    for epoch in range(1, n_epochs + 1):
        # Train loop
        for i, (X_batch, y_batch) in enumerate(train_ds):
            X_batch, y_batch = jnp.array(X_batch), jnp.array(y_batch)
            params, state, opt_state, loss, metric = update(
                params, state, key, X_batch, y_batch, optimizer, opt_state
            )
            evals(loss, metric, None, None)
            if verbose:
                print(f'  batch {i} - loss: {loss:.4f} - metric: {metric:.4f}  ',
                      end='\r')

        # Validation Loop
        for X_batch, y_batch in val_ds:
            X_batch, y_batch = jnp.array(X_batch), jnp.array(y_batch)
            loss, (state, metric) = evaluate(
                params, state, key, X_batch, y_batch, is_training=False
            )
            evals(None, None, loss, metric)

        evaluations = evals.values
        for i in range(4):
            evals_epoch[i].append(evaluations[i])
        evals.reset()

        if verbose:
            print(f'\n\nEpoch {epoch}/{n_epochs}\n'
                  f'    train_loss: {evaluations[0]:.4f} - '
                  f'train_metric: {evaluations[1]:.4f}\n'
                  f'    val_loss: {evaluations[2]:.4f} - '
                  f'val_metric: {evaluations[3]:.4f}')

    return tuple([params, state, opt_state, optimizer, *evals_epoch])


def test_loop(params, state, key, test_ds):
    evals = MultiMean(2)
    for X_batch, y_batch in test_ds:
        X_batch, y_batch = jnp.array(X_batch), jnp.array(y_batch)
        loss, (state, metric) = evaluate(
            params, state, key, X_batch, y_batch, is_training=False
        )
        evals(loss, metric)
    return evals.values


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
    params, state, _, _, *evals_epoch = train_validation_loop(
        key, train_ds, val_ds, n_epochs=n_epochs, lr=lr, verbose=verbose
        )
    plt_curves(*evals_epoch)


    if save_name is not None and save_name != '':
        # Save model
        model_path = os.path.join('./models', save_name)
        save_jax_model(params, state, model_path)
        # Load it to test the saved model
        params, state = load_jax_model(model_path)

    # Test loop
    if test:
        evals = test_loop(params, state, key, test_ds)
        plt_curves_test(*evals)

    # Show the plots
    plt.show()
