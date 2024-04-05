from typing import Dict, List, Generator, Tuple, Optional
import tensorflow as tf
import numpy as np
import os 
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import tqdm
import PIL 

from pydantic import BaseModel

DESCRIPTOR_FILE = "horizons.json"

SCHEMA = {
    'height': tf.io.FixedLenFeature([1], dtype=tf.int64),
    'width': tf.io.FixedLenFeature([1], dtype=tf.int64),
    'depth': tf.io.FixedLenFeature([1], dtype=tf.int64),
    "raw_image": tf.io.FixedLenFeature([], dtype=tf.string),
    "horizon": tf.io.FixedLenFeature([6], dtype=tf.float32)
}

Path = mpath.Path

class Point(BaseModel):
    x: float 
    y: float

    def scaled_tuple(self, width: float = 1.0, height: float = 1.0) -> Tuple[float, float]:
        return (self.x * width / 100.0, self.y * height / 100.0)

def _bytes_feature(value) -> tf.train.Feature:
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value: float) -> tf.train.Feature:
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value: int) -> tf.train.Feature:
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class Horizon(BaseModel):
    path: str
    start: Point
    control: Point
    end: Point 


    def load(self) -> PIL.Image:
        return PIL.Image.open(self.path)

    def to_norm_tensor(self) -> np.ndarray:
        return self.to_tensor(1.0, 1.0)

    def to_list(self, width: float = 1.0, height: float = 1.0) -> List[float]:
        sx, sy = self.start.scaled_tuple(width=width, height=height)
        cx, cy = self.control.scaled_tuple(width=width, height=height)
        ex, ey = self.end.scaled_tuple(width=width, height=height)    
        return [sx, sy, cx, cy, ex, ey]
        
    def to_tensor(self, width: float, height: float) -> np.ndarray:
        return np.asarray(self.to_list(width=width, height=height))

    def bezier(self, width: float, height: float) -> mpatches.PathPatch:
        return mpatches.PathPatch(
            Path(
                [
                    self.start.scaled_tuple(width=width, height=height), 
                    self.control.scaled_tuple(width=width, height=height), 
                    self.end.scaled_tuple(width=width, height=height)
                ],
                [Path.MOVETO, Path.CURVE3, Path.CURVE3],
            ),
            facecolor="none",
            lw=2,
        )

    def to_example(self) -> tf.train.Example:
        image = tf.io.read_file(self.path)
        image_shape = tf.io.decode_image(image).shape

        feature = {
            'height': _int64_feature(image_shape[0]),
            'width': _int64_feature(image_shape[1]),
            'depth': _int64_feature(image_shape[2]),
            "raw_image": _bytes_feature(image),
            "horizon": tf.train.Feature(float_list=tf.train.FloatList(value=self.to_list()))
        }
        
        return tf.train.Example(features=tf.train.Features(feature=feature))


def descriptors() -> Generator[Horizon, None, None]: 
    with open(DESCRIPTOR_FILE, "r") as f:
        data = json.load(f)
    
    for path, raw in data.items():
        yield Horizon(path=os.path.join("images", path), **raw)




def write_tf_record():
    horizons = descriptors()

    chunk = 0
    while True:
        written = 0
        dst = f"tfrecords/chunk_{chunk}.tfrecord"
        with tf.io.TFRecordWriter(dst) as writer:
            for horiz in horizons:
                example = horiz.to_example()
                s = example.SerializeToString()
                writer.write(s)

                written += len(s)
                if written > (100 * 1024 * 1024):
                    break
        
        if written == 0:
            os.remove(dst)
            break
        chunk += 1



def get_tfrecord_files() -> List[str]:
    return [f"tfrecords/{f}" for f in os.listdir("tfrecords")]


def load_datasets(
    files: Optional[List[str]] = None, 
    batch_size: int = 8, 
) -> tf.data.Dataset:
    @tf.py_function(Tout=[tf.float32, tf.float32])
    def decode(s): 
        rec = tf.io.parse_single_example(s, SCHEMA)
        image = tf.io.decode_image(rec["raw_image"])
        image = tf.image.convert_image_dtype(image, tf.float32)
        horizon = rec["horizon"]
        return (image, horizon)

    if files is None:
        files = get_tfrecord_files()

    return (
        tf.data.TFRecordDataset(files)
            .map(decode)
            .batch(batch_size=batch_size)
            .prefetch(tf.data.AUTOTUNE)
    )


if __name__ == "__main__":
    write_tf_record()