import tensorflow as tf 
from horizon.model import get_network
from horizon.dataset import load_datasets


def run():
    ds: tf.data.Dataset = load_datasets()
    model: tf.keras.Model = get_network()
    
    model.compile(loss="mse")
    model.fit(ds)


if __name__ == "__main__":
    run()