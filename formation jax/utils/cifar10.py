import os
from jax import numpy as jnp
import tensorflow as tf


def preprocessing(X, y):
    X = tf.cast(X, tf.float32) / 255.0
    y = tf.one_hot(y, 10, dtype=tf.float32)
    y = tf.reshape(y, [-1, 10])
    return X, y


def load_cifar10(batch_size=64, val_prop=0.2, test_prop=0.1, seed=0):
    def prepare_dataset(ds):
        ds = ds.shuffle(buffer_size=len(ds))
        ds = ds.batch(batch_size).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
            ).map(preprocessing)
        return ds

    tf.random.set_seed(seed)
    data = tf.keras.datasets.cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = data
    rd_perm = tf.random.shuffle(tf.range(len(x_train) + len(x_test)))
    x = tf.gather(tf.concat([x_train, x_test], axis=0), rd_perm, axis=0)
    y = tf.gather(tf.concat([y_train, y_test], axis=0), rd_perm, axis=0)
    full_ds = tf.data.Dataset.from_tensor_slices((x, y))
    len_full = len(full_ds)
    len_val = int(len_full * val_prop)
    len_test = int(len_full * test_prop)
    len_train = len_full - len_val - len_test

    train_ds = full_ds.take(len_train)
    val_ds = full_ds.skip(len_train).take(len_val)
    test_ds = full_ds.skip(len_train + len_val)

    train_ds = prepare_dataset(train_ds)
    val_ds = prepare_dataset(val_ds)
    test_ds = prepare_dataset(test_ds)

    print('training num batchs:', len(train_ds))
    print('validation num batchs:', len(val_ds))
    print('test num batchs:', len(test_ds))

    return train_ds, val_ds, test_ds
