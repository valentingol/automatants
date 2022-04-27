from tabnanny import verbose
from typing import Dict, Iterable

import numpy as np
import tensorflow as tf
import wandb

from train_rnn import get_data, LSTM

kl = tf.keras.layers
Optimizer = tf.keras.optimizers.Optimizer

def fit(model: tf.Module, optimizer: Optimizer, train_data: Iterable,
        valid_data: Iterable, epochs: int, verbose: bool = True
        ) -> Dict[str, np.ndarray]:
    """ Fit a RNN model on a dataset. """
    @tf.function
    def train_model(batch: tf.Tensor):
        X, Y = batch[..., :-1], batch[..., -1:]
        with tf.GradientTape() as tape:
            Y_pred = model(X, training=True)
            loss = tf.reduce_mean(tf.square(Y_pred - Y))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def valid_model(batch):
        X, Y = batch[..., :-1], batch[..., -1:]
        Y_pred = model(X, training=False)
        loss = tf.reduce_mean(tf.square(Y_pred - Y))
        return loss

    history = {'train_loss': [], 'val_loss': []}
    for epoch in range(epochs):
        if verbose:
            print(f'Epoch {epoch + 1}/{epochs} -', end=' ')
        train_losses = []
        for batch in train_data:
            loss = train_model(batch)
            train_losses.append(loss.numpy())
        if 'reset_states' in dir(model):
            # Reset states of the model between epochs if it is a custom RNN
            model.reset_states()
        mean_train_loss = np.mean(train_losses)
        history['train_loss'].append(mean_train_loss)
        if verbose:
            print(f'train loss:{mean_train_loss: .3f} -', end=' ')

        valid_losses = []
        for batch in valid_data:
            valid_model(batch)
            valid_losses.append(loss.numpy())
        model.reset_states()
        mean_val_loss = np.mean(valid_losses)
        history['val_loss'].append(mean_val_loss)
        if verbose:
            print(f'val loss:{mean_val_loss: .3f}')

        # wandb logging
        wandb.log({'train_loss': mean_train_loss, 'val_loss': mean_val_loss})

    return history


def run():
    default_config = {
        'batch_size': 10,
        'seq_len': 67,
        'learning_rate': 3e-4,
        'epochs': 100,
        'units': 10,
        'verbose': False,
        }
    wandb.init(project='rnn-wandb', config=default_config)
    config = wandb.config
    # Dataset
    train_valid_data = get_data('data/ETTh1.csv', config.batch_size,
                                config.seq_len)
    train_data = train_valid_data[20:]
    valid_data = train_valid_data[20:]
    dim = train_data.shape[-1]

    # Model and optimizer
    units = config.units
    LSTMClass = LSTM if custom_model else kl.LSTM
    model = tf.keras.Sequential([
        LSTMClass(units, return_sequences=True, stateful=True,
                batch_input_shape=(config.batch_size, config.seq_len, dim - 1)),
        kl.Dense(units, activation='relu'),
        kl.Dense(1)
        ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(optimizer=optimizer)
    fit(model, optimizer, train_data, valid_data, config.epochs, config.verbose)
    wandb.finish(quiet = True)

if __name__ == '__main__':
    # Configs
    # 17420 = 26 × 10 × 67
    custom_model = False
    sweep_config = {
        'project': 'rnn-wandb',
        'sweep': {
            'method': 'bayes',
            'metric': {'name': 'val_loss', 'goal': 'minimize'},
            'parameters': {
                'learning_rate': {
                    'distribution': 'log_uniform_values',
                    'min': 0.001,
                    'max': 0.3,
                    },
                'units': {
                    'distribution': 'int_uniform',
                    'min': 5,
                    'max': 64,
                    },
                }
            }
        }
    sweep_id = wandb.sweep(**sweep_config)
    wandb.agent(sweep_id, function=run)
