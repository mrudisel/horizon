
IMAGE_SHAPE = [1080, 1920, 3]

SCHEMA = {
    'height': tf.io.FixedLenFeature([1], dtype=tf.int64),
    'width': tf.io.FixedLenFeature([1], dtype=tf.int64),
    'depth': tf.io.FixedLenFeature([1], dtype=tf.int64),
    "raw_image": tf.io.FixedLenFeature([], dtype=tf.string),
    "horizon": tf.io.FixedLenFeature([6], dtype=tf.float32)
}


def get_network():
    images = tf.keras.Input(shape=IMAGE_SHAPE)
    x = tf.keras.layers.Conv2D(3)(images)
    x = tf.keras.layers.Flatten(dtype=tf.float32)(x)
    x = tf.keras.layers.Dense(6)(x)
    return tf.keras.Model(inputs=images, outputs=x)


