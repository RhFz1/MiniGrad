#!/usr/bin/env python
import numpy as np
import tinygrad.optim as optim
from tinygrad.tensor import Tensor
from tinygrad.utils import layer_init_uniform, fetch_mnist
from tqdm import trange

X_train, Y_train, X_test, Y_test = fetch_mnist()
class TinyBobNet:
  def __init__(self):
    self.l1 = Tensor(layer_init_uniform(784, 128))
    self.l2 = Tensor(layer_init_uniform(128, 10))

  def forward(self, x):
    return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

# model and optimizer

model = TinyBobNet()
#optim = optim.SGD([model.l1, model.l2], lr=0.01)
optim = optim.Adam([model.l1, model.l2], lr=3e-4)

BS = 128
losses, accuracies = [], []
for i in (t := trange(1000)):
  samp = np.random.randint(0, X_train.shape[0], size=(BS))
  
  x = Tensor(X_train[samp].reshape((-1, 28*28)))
  Y = Y_train[samp]
  y = np.zeros((len(samp),10), np.float32)
  y[range(y.shape[0]),Y] = -10.0
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