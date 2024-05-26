from typing import Tuple 
import tensorflow as tf 

tf.debugging.disable_traceback_filtering()

from horizon.model import get_network
from horizon.dataset import load_datasets, SAMPLES


TRAIN_SPLIT = 0.8

BATCH_SIZE = 8

VAL_SPLIT = 1 - TRAIN_SPLIT


def run():
    pair: Tuple[tf.data.Dataset, Tuple[int, int, int]] = load_datasets()
    
    ds, image_shape = pair

    ds = ds.shuffle(100)
        
    train_samples = int(SAMPLES * TRAIN_SPLIT)

    train_ds = ds.take(train_samples).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = ds.skip(train_samples).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model: tf.keras.Model = get_network(
        shape=image_shape, 
        block_depth=4,
    )


    checkpoint = tf.keras.callbacks.ModelCheckpoint("weights")

    model.compile(loss="mse", optimizer="adam")

    model.summary()

    model.fit(train_ds, validation_data=val_ds, callbacks=[checkpoint], epochs=10)


if __name__ == "__main__":
    run()