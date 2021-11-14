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

def tf_to_jax(tf_tensor1, tf_tensor2):
    jax_tensor1 = jnp.array(tf_tensor1)
    jax_tensor2 = jnp.array(tf_tensor2)
    return jax_tensor1, jax_tensor2


class VGG8(hk.Module):
    def __init__(self, name='VGG'):
        super().__init__(name=name)
        self.conv1_1 = hk.Conv2D(32, 3, w_init=self.init_he_normal(32))
        self.bn1_1 = hk.BatchNorm(True, True, decay_rate=0.99)
        self.conv1_2 = hk.Conv2D(32, 3, w_init=self.init_he_normal(32))
        self.bn1_2 = hk.BatchNorm(True, True, 0.99)


        self.conv2_1 = hk.Conv2D(64, 3, w_init=self.init_he_normal(64))
        self.bn2_1 = hk.BatchNorm(True, True, 0.99)
        self.conv2_2 = hk.Conv2D(64, 3, w_init=self.init_he_normal(64))
        self.bn2_2 = hk.BatchNorm(True, True, 0.99)

        self.conv3_1 = hk.Conv2D(128, 3, w_init=self.init_he_normal(128))
        self.bn3_1 = hk.BatchNorm(True, True, 0.99)
        self.conv3_2 = hk.Conv2D(128, 3, w_init=self.init_he_normal(128))
        self.bn3_2 = hk.BatchNorm(True, True, 0.99)

        self.linear1 = hk.Linear(128, w_init=self.init_he_normal(128))
        self.bn4 = hk.BatchNorm(True, True, 0.99)
        self.linear2 = hk.Linear(10, w_init=self.init_glorot_normal(128, 10))

    def init_he_normal(self, n_out):
        std = jnp.sqrt(2.0 / n_out)
        return hk.initializers.RandomNormal(std, 0.0)

    def init_glorot_normal(self, n_in, n_out):
        std = jnp.sqrt(2.0 / (n_in + n_out))
        return hk.initializers.RandomNormal(std, 0.0)

    def forward(self, X, training=True):
        # Block 1
        X = jax.nn.relu(self.conv1_1(X))
        X = self.bn1_1(X, is_training=training)
        X = jax.nn.relu(self.conv1_2(X))
        X = self.bn1_2(X, is_training=training)
        X = hk.max_pool(X, [2, 2], 2, 'SAME')
        if training:
            X = hk.dropout(hk.next_rng_key(), 0.2, X)

        # Block 2
        X = jax.nn.relu(self.conv2_1(X))
        X = self.bn2_1(X, is_training=training)
        X = jax.nn.relu(self.conv2_2(X))
        X = self.bn2_2(X, is_training=training)
        X = hk.max_pool(X, [2, 2], 2, 'SAME')
        if training:
            X = hk.dropout(hk.next_rng_key(), 0.3, X)

        # Block 3
        X = jax.nn.relu(self.conv3_1(X))
        X = self.bn3_1(X, is_training=training)
        X = jax.nn.relu(self.conv3_2(X))
        X = self.bn3_2(X, is_training=training)
        X = hk.max_pool(X, [2, 2], 2, 'SAME')
        if training:
            X = hk.dropout(hk.next_rng_key(), 0.4, X)

        # Classifier
        X = hk.Flatten()(X)
        X = jax.nn.relu(self.linear1(X))
        X = self.bn4(X, is_training=training)
        if training:
            X = hk.dropout(hk.next_rng_key(), 0.5, X)
        X = self.linear2(X)
        return X

    def __call__(self, X, training=True):
        return self.forward(X, training)


@hk.transform_with_state
def forward_pass(X, training):
    vgg = VGG8()
    y_pred = vgg(X, training)
    return y_pred


@partial(jit, static_argnums=(4, ))
def evaluate(params, state, X, y, training, key):
    y_pred, state = forward_pass.apply(params, state, key, X, training)
    loss = (jit(optax.softmax_cross_entropy)(y_pred, y)).mean()
    metric = (jnp.argmax(y_pred, axis=-1)
              == jnp.argmax(y, axis=-1)).astype(jnp.float32).mean()
    return loss, (loss, metric, state)


def init_states(key, ds, lr):
    for X, y in ds:
        X, _ = tf_to_jax(X, y)
        params, state = forward_pass.init(key, X, True)
        break
    optimizer = optax.sgd(lr, momentum=0.9)
    opt_state = optimizer.init(params)
    return params, optimizer, opt_state, state


@partial(jit, static_argnums=(3, ))
def train(params, state, opt_state, optimizer, X, y, key):
    grads, (train_loss, train_metric, state) = grad(evaluate, has_aux=True)(
        params, state, X, y, True, key
        )
    update, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, update)
    return params, state, opt_state, train_loss, train_metric


def validate(params, state, X, y, key):
    val_loss, (_, val_metric, state) = evaluate(
        params, state, X, y, False, key
        )
    return state, val_loss, val_metric


def train_loop(train_ds, params, state, opt_state, optimizer, key):
    train_loss_mean, train_metric_mean = jnp.zeros(1), jnp.zeros(1)
    i = 0
    for X_batch, y_batch in train_ds:
        X_batch, y_batch = tf_to_jax(X_batch, y_batch)
        params, state, opt_state, train_loss, train_metric = train(
            params, state, opt_state, optimizer, X_batch, y_batch, key
            )
        train_loss_mean += train_loss
        train_metric_mean += train_metric
        i += 1
    train_loss_mean = train_loss_mean / i
    train_metric_mean = train_metric_mean / i
    return params, state, opt_state, train_loss_mean, train_metric_mean


def val_loop(val_ds, params, state, key):
    val_loss_mean, val_metric_mean = jnp.zeros(1), jnp.zeros(1)
    i = 0
    for X_batch, y_batch in val_ds:
        X_batch, y_batch = tf_to_jax(X_batch, y_batch)
        state, val_loss, val_metric = validate(
            params, state, X_batch, y_batch, key
            )
        val_loss_mean += val_loss
        val_metric_mean += val_metric
        i += 1
    val_loss_mean = val_loss_mean / i
    val_metric_mean = val_metric_mean / i
    return val_loss_mean, val_metric_mean


def test_loop(test_ds, params, state, key):
    test_loss, test_metric = val_loop(test_ds, params, state, key)
    test_loss, test_metric = test_loss.item(), test_metric.item()
    return test_loss, test_metric


def train_valid_loop(train_ds, val_ds, n_epochs, key, lr, verbose=True):
    # Initialization
    params, optimizer, opt_state, state = init_states(key, train_ds, lr)
    print('initialization done')
    train_losses, val_losses = [], []
    train_metrics, val_metrics = [], []
    for epoch in range(n_epochs):
        # Training loop
        params, state, opt_state, train_loss, train_metric = train_loop(
            train_ds, params, state, opt_state, optimizer, key
            )
        # Validation loop
        val_loss, val_metric = val_loop(val_ds, params, state, key)

        train_losses.append(train_loss.item())
        train_metrics.append(train_metric.item())
        val_losses.append(val_loss.item())
        val_metrics.append(val_metric.item())

        if verbose:
            print(f'Epoch {epoch + 1}/{n_epochs}\n'
                f'  train loss: {train_loss.item(): .4f} '
                f'  train metric: {train_metric.item(): .4f}\n'
                f'  valid loss {val_loss.item(): .4f} '
                f'  valid metric {val_metric.item(): .4f}')
    if verbose:
        print()
    return (params, state, opt_state, train_losses, train_metrics,
            val_losses, val_metrics)


if __name__ == '__main__':
    # Configs
    save_name = 'My_vgg8' # None or empty to not save
    seed = 0
    n_epochs = 300
    batch_size = 64
    val_prop = 0.08
    test_prop = 0.16
    lr = 0.01

    key = random.PRNGKey(seed)

    train_ds, val_ds, test_ds = load_cifar10(batch_size=batch_size,
                                             val_prop=val_prop,
                                             test_prop=test_prop,
                                             seed=seed)

    (params, state, opt_state, train_losses, train_metrics,
     val_losses, val_metrics) = train_valid_loop(
        train_ds, val_ds, n_epochs, key, lr, verbose=True
        )
    plt_curves(train_losses, train_metrics, val_metrics, val_losses)


    if save_name is not None and save_name != '':
        # Save model
        model_path = os.path.join('./models', save_name)
        save_jax_model(params, state, model_path)
        # Load it to test the saved model
        params, state = load_jax_model(model_path)

    # Test loop (comment out if you don't want to test)
    test_loss, test_metric = test_loop(test_ds, params, state, key)
    plt_curves_test(test_loss, test_metric)

    # Show the plots
    plt.show()
