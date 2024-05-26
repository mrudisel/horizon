import tensorflow as tf
tf.debugging.disable_traceback_filtering()

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



def get_network(
    shape=[1080, 1920, 3], 
    block_depth=2, 
    conv_units=32,
    pool_size=2,
    pool_every=1,
):
    images = tf.keras.Input(shape=shape)

    x = tf.keras.layers.Conv2D(conv_units, kernel_size=3, activation="relu")(images)
    x = tf.keras.layers.AveragePooling2D(pool_size=pool_size)(x)

    # skips = []
    # for width in widths[:-1]:
    #     x = DownBlock(width, block_depth)([x, skips])

    # for _ in range(block_depth):
    #    x = ResidualBlock(widths[-1])(x)

    # for width in reversed(widths[:-1]):
    #     x = UpBlock(width, block_depth)([x, skips])

    for i in range(block_depth):
        resid = x 
        x = tf.keras.layers.BatchNormalization(center=False, scale=False)(x)
        x = tf.keras.layers.Conv2D(
            conv_units, kernel_size=3, padding="same", activation=tf.keras.activations.swish
        )(x)
        x = tf.keras.layers.Conv2D(conv_units, kernel_size=3, padding="same")(x)
        x = tf.keras.layers.Add()([x, resid])
        
        if i % pool_every == 0:
            x = tf.keras.layers.AveragePooling2D(pool_size=pool_size)(x)


    x = tf.keras.layers.Conv2D(conv_units / 2, kernel_size=3, padding="same")(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=pool_size)(x)
    x = tf.keras.layers.Conv2D(conv_units / 4, kernel_size=3, padding="same")(x)

    if (block_depth - 1) % pool_every != 0:
        x = tf.keras.layers.AveragePooling2D(pool_size=pool_size)(x)
    
    x = tf.keras.layers.Flatten(dtype=tf.float32)(x)

    x = tf.keras.layers.Dense(128, activation="relu", kernel_initializer="zeros")(x)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_initializer="zeros")(x)
    x = tf.keras.layers.Dense(32, activation="relu", kernel_initializer="zeros")(x)
    x = tf.keras.layers.Dense(6, activation="relu", kernel_initializer="zeros")(x)

    return tf.keras.Model(images, x, name="unet")