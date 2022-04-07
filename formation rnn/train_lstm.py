import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

def get_data(csv_path, batch_size, seq_len):
    df = pd.read_csv(csv_path)
    df = df.drop(columns='date')
    df = (df - df.mean()) / df.std()
    data = tf.constant(df.values)
    data = tf.cast(data, tf.float32)
    dim = data.shape[-1]

    n_batch = len(data) // batch_size // seq_len  # 26
    batchs = [tf.zeros((1, 0, seq_len, dim)) for _ in range(n_batch)]
    for chunk in range(batch_size):
        for seq in range(n_batch):
            start = (chunk * n_batch + seq) * seq_len
            sequence = data[start: start + seq_len]
            sequence = tf.expand_dims(sequence, axis=0)
            sequence = tf.expand_dims(sequence, axis=0)
            batchs[seq] = tf.concat([batchs[seq], sequence], axis=1)
    data = tf.concat(batchs, axis=0)
    return data


def fit(model, optimizer, train_data, valid_data, epochs):
    history = {'train_loss': [], 'val_loss': []}
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs} -', end=' ')
        train_losses = []
        for batch in train_data:
            X, Y = batch[..., :-1], batch[..., -1:]
            with tf.GradientTape() as tape:
                Y_pred = model(X, training=True)
                loss = tf.reduce_mean(tf.square(Y_pred - Y))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_losses.append(loss.numpy())
        mean_train_loss = np.mean(train_losses)
        history['train_loss'].append(mean_train_loss)
        print(f'train loss:{mean_train_loss: .3f} -', end=' ')

        valid_losses = []
        for batch in valid_data:
            X, Y = batch[..., :-1], batch[..., -1:]
            Y_pred = model(X, training=False)
            loss = tf.reduce_mean(tf.square(Y_pred - Y))
            valid_losses.append(loss.numpy())
        mean_val_loss = np.mean(valid_losses)
        history['val_loss'].append(mean_val_loss)
        print(f'val loss:{mean_val_loss: .3f}')
    return history


class LSTM(tf.keras.Model):
    def __init__(self, units: int, batch_input_shape:tuple[int], activation:str='tanh',
                 rec_activation:str='sigmoid', return_sequences:bool=False,
                 return_state:bool=False, stateful:bool=False, ):
        super().__init__()
        self.units = units
        self.activation = self.get_activation(activation)
        self.rec_activation = self.get_activation(rec_activation)
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.stateful = stateful
        self.batch_input_shape = batch_input_shape
        input_dim = batch_input_shape[-1]
        # Input gate
        self.IG_w = tf.Variable(tf.random.normal((units + input_dim, units)), trainable=True)
        self.IG_b = tf.Variable(tf.random.normal((units, )), trainable=True)
        # Output gate
        self.OG_w = tf.Variable(tf.random.normal((units + input_dim, units)), trainable=True)
        self.OG_b = tf.Variable(tf.random.normal((units, )), trainable=True)
        # Forget Gate
        self.FG_w = tf.Variable(tf.random.normal((units + input_dim, units)), trainable=True)
        self.FG_b = tf.Variable(tf.random.normal((units, )), trainable=True)
        # Cell Gate
        self.CG_w = tf.Variable(tf.random.normal((units + input_dim, units)), trainable=True)
        self.CG_b = tf.Variable(tf.random.normal((units, )), trainable=True)
        # States
        self.h = tf.zeros((batch_input_shape[0], units))
        self.c = tf.zeros((batch_input_shape[0], units))

    def get_activation(self, activation_name):
        if activation_name == 'tanh':
            return tf.keras.activations.tanh
        elif activation_name == 'sigmoid':
            return tf.keras.activations.sigmoid
        elif activation_name == 'relu':
            return tf.keras.activations.relu
        elif activation_name == 'selu':
            return tf.keras.activations.selu
        elif activation_name == 'linear':
            return tf.keras.activations.linear
        else:
            raise ValueError('Activation function not found')

    def call(self, seq):
        if not self.stateful:
            self.h = tf.zeros((self.batch_input_shape[0], self.units))
            self.c = tf.zeros((self.batch_input_shape[0], self.units))
        h, c = self.h, self.c
        seq_h, seq_c = [], []
        for i in range(tf.shape(seq)[1]):
            x = seq[:, i, :]
            xh = tf.concat([x, h], axis=-1)
            input_xh = self.rec_activation(xh @ self.IG_w + self.IG_b)
            forget_xh = self.rec_activation(xh @ self.FG_w + self.FG_b)
            output_xh = self.rec_activation(xh @ self.OG_w + self.OG_b)
            new_c = self.activation(xh @ self.CG_w + self.CG_b)
            c = c * forget_xh + new_c * input_xh
            h = output_xh * self.activation(c)
            seq_h.append(tf.expand_dims(h, axis=1))
            seq_c.append(tf.expand_dims(c, axis=1))
        seq_h = tf.concat(seq_h, axis=1)
        seq_c = tf.concat(seq_c, axis=1)
        self.h, self.c = h, c
        if self.return_sequences:
            if self.return_state:
                return seq_h, seq_c
            else:
                return seq_h
        else:
            if self.return_state:
                return h, c
            else:
                return h


if __name__ == '__main__':
    kl = tf.keras.layers
    # 17420 = 26 × 10 × 67
    batch_size = 10
    seq_len = 67
    learning_rate = 3e-4
    epochs = 45

    train_valid_data = get_data('data/ETTh1.csv', batch_size, seq_len)
    train_data = train_valid_data[:20]
    valid_data = train_valid_data[20:]
    dim = train_data.shape[-1]

    model = tf.keras.Sequential([
        LSTM(64, return_sequences=True, stateful=True,
                batch_input_shape=(batch_size, seq_len, dim - 1)),
        kl.Dense(64, activation='relu'),
        kl.Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer)
    history = fit(model, optimizer, train_data, valid_data, epochs)

    plt.title('Losses')
    plt.plot(history['train_loss'], c='#ff7f17', label='train')
    plt.plot(history['val_loss'], c='#0588ed', label='validation')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

    forecast_data = get_data('data/ETTh1.csv', 1, 260)[0]
    model_forecast = tf.keras.Sequential([
        kl.LSTM(64, return_sequences=True,
                batch_input_shape=(None, None, dim - 1)),
        kl.Dense(64, activation='relu'),
        kl.Dense(1)
    ])
    model_forecast.set_weights(model.get_weights())
    target = tf.squeeze(forecast_data[..., -1])
    pred = model_forecast(forecast_data[..., :-1])
    pred = tf.squeeze(pred)

    plt.title('Forecasting')
    plt.plot(pred.numpy(), c='#ff7f17', label='prediction')
    plt.plot(target.numpy(), c='#0588ed', label='ground truth')
    plt.xlabel('time')
    plt.legend()
    plt.show()
