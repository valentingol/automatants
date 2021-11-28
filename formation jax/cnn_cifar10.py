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


def tf_to_jax(tf_tensor1, tf_tensor2):
    jax_tensor1 = jnp.array(tf_tensor1)
    jax_tensor2 = jnp.array(tf_tensor2)
    return jax_tensor1, jax_tensor2


class VGG8(hk.Module):
    def __init__(self, name='VGG'):
        super().__init__(name=name)
        # Vscaling(2.0) = He normal
        self.conv1_1 = hk.Conv2D(32, 3, w_init=Vscaling(2.0))
        # BatchNorm : create_scale, create_offset, decay_ema
        self.bn1_1 = hk.BatchNorm(True, True, decay_rate=0.99)
        self.conv1_2 = hk.Conv2D(32, 3, w_init=Vscaling(2.0))
        self.bn1_2 = hk.BatchNorm(True, True, 0.99)


        self.conv2_1 = hk.Conv2D(64, 3, w_init=Vscaling(2.0))
        self.bn2_1 = hk.BatchNorm(True, True, 0.99)
        self.conv2_2 = hk.Conv2D(64, 3, w_init=Vscaling(2.0))
        self.bn2_2 = hk.BatchNorm(True, True, 0.99)

        self.conv3_1 = hk.Conv2D(128, 3, w_init=Vscaling(2.0))
        self.bn3_1 = hk.BatchNorm(True, True, 0.99)
        self.conv3_2 = hk.Conv2D(128, 3, w_init=Vscaling(2.0))
        self.bn3_2 = hk.BatchNorm(True, True, 0.99)

        self.linear1 = hk.Linear(128, w_init=Vscaling(2.0))
        self.bn4 = hk.BatchNorm(True, True, 0.99)
        # Vscaling(1.0, mode='fan_avg') = Glorot normal
        self.linear2 = hk.Linear(10, w_init=Vscaling(1.0, mode='fan_avg'))

    def forward(self, X, is_training=True):
        # Block 1
        X = jax.nn.relu(self.conv1_1(X))
        X = self.bn1_1(X, is_training=is_training)
        X = jax.nn.relu(self.conv1_2(X))
        X = self.bn1_2(X, is_training=is_training)
        X = hk.max_pool(X, [2, 2], 2, 'SAME')
        if is_training:
            X = hk.dropout(hk.next_rng_key(), 0.2, X)

        # Block 2
        X = jax.nn.relu(self.conv2_1(X))
        X = self.bn2_1(X, is_training=is_training)
        X = jax.nn.relu(self.conv2_2(X))
        X = self.bn2_2(X, is_training=is_training)
        X = hk.max_pool(X, [2, 2], 2, 'SAME')
        if is_training:
            X = hk.dropout(hk.next_rng_key(), 0.3, X)

        # Block 3
        X = jax.nn.relu(self.conv3_1(X))
        X = self.bn3_1(X, is_training=is_training)
        X = jax.nn.relu(self.conv3_2(X))
        X = self.bn3_2(X, is_training=is_training)
        X = hk.max_pool(X, [2, 2], 2, 'SAME')
        if is_training:
            X = hk.dropout(hk.next_rng_key(), 0.4, X)

        # Classifier
        X = hk.Flatten()(X)
        X = jax.nn.relu(self.linear1(X))
        X = self.bn4(X, is_training=is_training)
        if is_training:
            X = hk.dropout(hk.next_rng_key(), 0.5, X)
        X = self.linear2(X)
        return X

    def __call__(self, X, is_training=True):
        return self.forward(X, is_training)


@hk.transform_with_state
def forward_pass(X, is_training):
    vgg = VGG8()
    y_pred = vgg(X, is_training)
    return y_pred


def init_states(key, ds, lr):
    for X, y in ds:
        X, _ = tf_to_jax(X, y)
        params, state = forward_pass.init(key, X, True)
        break
    optimizer = optax.sgd(lr, momentum=0.9)
    opt_state = optimizer.init(params)
    return params, optimizer, opt_state, state


def evaluate(params, state, key, X, y, is_training):
    y_pred, state = forward_pass.apply(params, state, key, X, is_training)
    loss = optax.softmax_cross_entropy(y_pred, y).mean()
    metric = (jnp.argmax(y_pred, axis=-1)
              == jnp.argmax(y, axis=-1)).mean()
    return loss, (metric, state)


@partial(jit, static_argnums=4)
def train(params, state, key, opt_state, optimizer, X, y):
    (train_loss, (train_metric, state)), grads = vgrad(evaluate, has_aux=True)(
        params, state, key, X, y, True,
        )
    update, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, update)
    return params, state, opt_state, train_loss, train_metric


@jit
def validate(params, state, key, X, y):
    val_loss, (val_metric, state) = evaluate(
        params, state, key, X, y, False,
        )
    return state, val_loss, val_metric


def train_valid_loop(key, train_ds, val_ds, n_epochs, lr,
                     verbose=True):
    # Initialization
    params, optimizer, opt_state, state = init_states(key, train_ds, lr)

    if verbose:
        print('Initialization done')
    # evals & epoch_evals: train_loss, train_metric, val_loss, val_metric
    evals = MultiMean(4)
    evals_epoch = [[] for _ in range(4)]
    for epoch in range(n_epochs):
        evals.reset()
        # Training loop
        for X_batch, y_batch in train_ds:
            X_batch, y_batch = tf_to_jax(X_batch, y_batch)
            params, state, opt_state, train_loss, train_metric = train(
                params, state, key, opt_state, optimizer, X_batch, y_batch
                )
            evals(train_loss, train_metric, None, None)

        # Validation loop
        for X_batch, y_batch in val_ds:
            X_batch, y_batch = tf_to_jax(X_batch, y_batch)
            state, val_loss, val_metric = validate(
                params, state, key, X_batch, y_batch
                )
            evals(None, None, val_loss, val_metric)

        train_loss, train_metric, val_loss, val_metric = evals.values

        evals_epoch[0].append(train_loss)
        evals_epoch[1].append(train_metric)
        evals_epoch[2].append(val_loss)
        evals_epoch[3].append(val_metric)

        if verbose:
            print(f'Epoch {epoch + 1}/{n_epochs}\n'
                f'  train loss: {train_loss: .4f} '
                f'  train metric: {train_metric: .4f}\n'
                f'  valid loss {val_loss: .4f} '
                f'  valid metric {val_metric: .4f}')
    if verbose:
        print()
    print(evals_epoch)
    return params, state, opt_state, *evals_epoch


def test_loop(key, test_ds, params, state):
    # evals: test_loss, test_metric
    evals = MultiMean(2)
    for X_batch, y_batch in test_ds:
        X_batch, y_batch = tf_to_jax(X_batch, y_batch)
        state, test_loss, test_metric = validate(
            params, state, key, X_batch, y_batch
            )
        evals(test_loss, test_metric)
    return state, *evals.values


if __name__ == '__main__':
    # Configs
    save_name = '' # None or empty to not save
    seed = 0
    n_epochs = 2
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

    (params, state, opt_state, train_losses, train_metrics,
     val_losses, val_metrics) = train_valid_loop(
        key, train_ds, val_ds, n_epochs, lr, verbose=True
        )
    plt_curves(train_losses, train_metrics, val_losses, val_metrics)


    if save_name is not None and save_name != '':
        # Save model
        model_path = os.path.join('./models', save_name)
        save_jax_model(params, state, model_path)
        # Load it to test the saved model
        params, state = load_jax_model(model_path)

    ## Test loop
    if test:
        state, test_loss, test_metric = test_loop(test_ds, params, state, key)
        plt_curves_test(test_loss, test_metric)

    # Show the plots
    plt.show()
