import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = tf.keras.layers.Conv2D(width, kernel_size=1)(x)
        x = tf.keras.layers.BatchNormalization(center=False, scale=False)(x)
        x = tf.keras.layers.Conv2D(
            width, kernel_size=3, padding="same", activation=tf.keras.activations.swish
        )(x)
        x = tf.keras.layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = tf.keras.layers.Add()([x, residual])
        return x

    return apply


def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = tf.keras.layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply



def get_network(shape=[1080, 1920, 3], widths=[64, 96, 128, 256], block_depth = 4):
    images = tf.keras.Input(shape=shape)

    x = tf.keras.layers.Conv2D(widths[0], kernel_size=1)(images)

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    x = tf.keras.layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros", dtype=tf.float32)(x)

    x = tf.keras.layers.Flatten(dtype=tf.float32)(x)

    x = tf.keras.layers.Dense(512, activation="relu", kernel_initializer="zeros")(x)
    x = tf.keras.layers.Dense(256, activation="relu", kernel_initializer="zeros")(x)
    x = tf.keras.layers.Dense(128, activation="relu", kernel_initializer="zeros")(x)
    x = tf.keras.layers.Dense(32, activation="relu", kernel_initializer="zeros")(x)
    x = tf.keras.layers.Dense(6, activation="relu", kernel_initializer="zeros")(x)

    return tf.keras.Model(images, x, name="unet")