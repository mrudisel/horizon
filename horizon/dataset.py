from typing import Dict, List, Generator, Tuple, Optional
import tensorflow as tf
tf.debugging.disable_traceback_filtering()

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

IMAGE_SHAPE = (1080, 1920, 3)

SAMPLES = 3768

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
        raw_image = tf.io.read_file(self.path)
        image = tf.io.decode_image(raw_image)

        feature = {
            'height': _int64_feature(image.shape[0]),
            'width': _int64_feature(image.shape[1]),
            'depth': _int64_feature(image.shape[2]),
            "raw_image": _bytes_feature(raw_image),
            "horizon": tf.train.Feature(float_list=tf.train.FloatList(value=self.to_list()))
        }
        
        return tf.train.Example(features=tf.train.Features(feature=feature))


def descriptors() -> Generator[Horizon, None, None]: 
    with open(DESCRIPTOR_FILE, "r") as f:
        data = json.load(f)
    
    for path, raw in data.items():
        yield Horizon(path=os.path.join("images", path), **raw)




def write_tf_record() -> int:
    horizons = descriptors()

    chunk = 0
    examples = 0
    while True:
        written = 0
        dst = f"tfrecords/chunk_{chunk}.tfrecord"
        with tf.io.TFRecordWriter(dst) as writer:
            for horiz in horizons:
                example = horiz.to_example()
                s = example.SerializeToString()
                writer.write(s)
                examples += 1

                written += len(s)
                if written > (100 * 1024 * 1024):
                    break
        
        if written == 0:
            os.remove(dst)
            break
        chunk += 1

    return examples



def get_tfrecord_files() -> List[str]:
    return [f"tfrecords/{f}" for f in os.listdir("tfrecords")]



def decode(s): 
    rec = tf.io.parse_single_example(s, SCHEMA)
    image = tf.io.decode_image(rec["raw_image"], expand_animations=False)
    horizon = rec["horizon"]

    tf.ensure_shape(image, IMAGE_SHAPE)
    tf.ensure_shape(horizon, [6])
    
    return (image, horizon)


def build_decode_resize(resize_shape: Tuple[int, int, int]): 
    shape = (resize_shape[0], resize_shape[1])
    def decode_resize(s):
        image, horizon = decode(s)
        image = tf.image.resize(image, shape, preserve_aspect_ratio=True)
        return (image, horizon)

    return decode_resize

def cast_images(img, horizon):
    img = tf.image.convert_image_dtype(img, tf.float32)
    return (img, horizon)


def load_datasets(
    files: Optional[List[str]] = None, 
    resize: float = 1,
) -> Tuple[tf.data.Dataset, Tuple[int, int, int]]:

    if files is None:
        files = get_tfrecord_files()

    if resize == 1:
        output_shape = IMAGE_SHAPE
        decode_fn = decode
    else:
        output_shape = (
            int(np.floor(IMAGE_SHAPE[0] * resize)),
            int(np.floor(IMAGE_SHAPE[1] * resize)),
            IMAGE_SHAPE[-1],
        )
        decode_fn = build_decode_resize(output_shape)
        print(f"resizing images to {output_shape}")


    ds = (
        tf.data.TFRecordDataset(files)
            .shuffle(len(files))
            .map(decode_fn)
            .map(cast_images)
    )

    return (ds, output_shape)


if __name__ == "__main__":
    examples = write_tf_record()
    print(f"wrote out {examples} examples")