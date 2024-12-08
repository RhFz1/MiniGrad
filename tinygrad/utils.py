import numpy as np
import os
import requests
import hashlib
import gzip
import numpy as np

def layer_init_uniform(fan_in, fan_out):
    ret = np.random.uniform(-1, 1, size=(fan_in, fan_out)) / np.sqrt(fan_in * fan_out)
    return ret.astype(np.float32)

def fetch_mnist():
# Function to download and load MNIST data
    def fetch_mnist_data(url, filename):
        # Ensure the data directory exists
        fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
        
        # Download the file if it doesn't exist
        if not os.path.exists(fp):
            print(f"Downloading {filename}...")
            data = requests.get(url, stream=True).content
            with open(fp, 'wb') as file:
                file.write(data)
        else:
            with open(fp, "rb") as f:
                data = f.read()
            return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()

    # URLs for the MNIST dataset
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz"
    }

    # Fetch the data
    X_train = fetch_mnist_data(base_url + files["train_images"], files["train_images"])[0x10:].reshape((-1, 28, 28))
    Y_train = fetch_mnist_data(base_url + files["train_labels"], files["train_labels"])[8:]
    X_test = fetch_mnist_data(base_url + files["test_images"], files["test_images"])[0x10:].reshape((-1, 28, 28))
    Y_test = fetch_mnist_data(base_url + files["test_labels"], files["test_labels"])[8:]
    
    return X_train, Y_train, X_test, Y_test