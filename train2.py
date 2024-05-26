
import tensorflow as tf

SCHEMA = {
    'height': tf.io.FixedLenFeature([1], dtype=tf.int64),
    'width': tf.io.FixedLenFeature([1], dtype=tf.int64),
    'depth': tf.io.FixedLenFeature([1], dtype=tf.int64),
    "raw_image": tf.io.FixedLenFeature([], dtype=tf.string),
    "horizon": tf.io.FixedLenFeature([6], dtype=tf.float32)
}

INPUT_SHAPE = (1080, 1920, 3)

def decode(s): 
    rec = tf.io.parse_single_example(s, SCHEMA)
    image = tf.io.decode_image(rec["raw_image"])
    image = tf.image.convert_image_dtype(image, tf.float32)
    horizon = rec["horizon"]
    return image, horizon

def get_dataset():
    ds = tf.data.TFRecordDataset(["tfrecords/chunk_31.tfrecord"])
    ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(2).repeat().prefetch(tf.data.AUTOTUNE)

def get_model():
    input_layer = tf.keras.Input(shape=INPUT_SHAPE, dtype=tf.float32)
    flat = tf.keras.layers.Flatten()(input_layer)
    output = tf.keras.layers.Dense(6, activation=None)(flat)
    model = tf.keras.Model(inputs=input_layer, outputs=output) 
    return model 

ds = get_dataset()
model = get_model()
model.compile(loss="mse", optimizer="adam") 

model.fit(ds, epochs=10, steps_per_epoch=3)