#!/usr/bin/env python
import os
import requests
import hashlib
import gzip
import numpy as np
from tinygrad.tensor import Tensor
from tqdm import trange

# Function to download and load MNIST data
def fetch_mnist(url, filename):
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
X_train = fetch_mnist(base_url + files["train_images"], files["train_images"])[0x10:].reshape((-1, 28, 28))
Y_train = fetch_mnist(base_url + files["train_labels"], files["train_labels"])[8:]
X_test = fetch_mnist(base_url + files["test_images"], files["test_images"])[0x10:].reshape((-1, 28, 28))
Y_test = fetch_mnist(base_url + files["test_labels"], files["test_labels"])[8:]

# train a model

def layer_init(m, h):
  ret = np.random.uniform(-1., 1., size=(m,h))/np.sqrt(m*h)
  return ret.astype(np.float32)

class TinyBobNet:
  def __init__(self):
    self.l1 = Tensor(layer_init(784, 128))
    self.l2 = Tensor(layer_init(128, 10))

  def forward(self, x):
    return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

# optimizer

class SGD:
  def __init__(self, tensors, lr):
    self.tensors = tensors
    self.lr = lr

  def step(self):
    for t in self.tensors:
      t.data -= self.lr * t.grad

model = TinyBobNet()
optim = SGD([model.l1, model.l2], lr=0.01)

BS = 128
losses, accuracies = [], []
for i in (t := trange(1000)):
  samp = np.random.randint(0, X_train.shape[0], size=(BS))
  
  x = Tensor(X_train[samp].reshape((-1, 28*28)))
  Y = Y_train[samp]
  y = np.zeros((len(samp),10), np.float32)
  y[range(y.shape[0]),Y] = -1.0
  y = Tensor(y)
  
  # network
  outs = model.forward(x)

  # NLL loss function
  loss = outs.mul(y).mean()
  loss.backward()
  optim.step()
  
  cat = np.argmax(outs.data, axis=1)
  accuracy = (cat == Y).mean()
  
  
  # printing
  loss = loss.data
  losses.append(loss)
  accuracies.append(accuracy)
  t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))

# evaluate
def numpy_eval():
  Y_test_preds_out = model.forward(Tensor(X_test.reshape((-1, 28*28))))
  Y_test_preds = np.argmax(Y_test_preds_out.data, axis=1)
  return (Y_test == Y_test_preds).mean()

accuracy = numpy_eval()
print("test set accuracy is %f" % accuracy)
assert accuracy > 0.95