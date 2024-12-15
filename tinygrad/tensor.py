# Trying to implement George Hotz's tinygrad in Python
from functools import partialmethod
import numpy as np

'''
    Understanding for reference
    Let's dive into the key components:

    Tensor Class
    The Tensor class is the fundamental building block of this computational framework. Think of it like a smart container for numerical data that can track its own computational history and compute gradients.

    Key attributes:

        data: The actual numerical data (a NumPy array)
        grad: Stores the gradient of this tensor
        _ctx: Stores the context of how this tensor was created (its computational origin)

    Key methods:

        __init__: Initializes a tensor, ensuring it's a NumPy array
        __str__: Provides a readable string representation
        backward(): The core gradient computation method

    If no context exists, does nothing
    If no gradient exists, initializes it (for scalar tensors)
    Recursively computes gradients by calling the backward method of parent operations

    Function Class
    The Function class represents a single computational operation (like addition, multiplication, ReLU). It's like a recipe for how to compute forward and backward passes.

    Key methods:

        __init__: Tracks parent tensors and saved tensors
        save_for_backward(): Stores tensors needed for gradient computation
        apply(): Applies the operation, creating a new tensor and tracking its context


        Registration Mechanism
        The register() function dynamically adds methods to the Tensor class. This allows you to call operations like tensor.add() or tensor.mul() directly.
'''


class Tensor:

    def __init__(self, data):

        if (type(data) is not np.ndarray):
            raise ValueError("Data must be of type numpy.ndarray", data)
            assert False
        self.data = data
        self.grad = None
        self.shape = data.shape
        # dont understand the use of this part
        self._ctx = None
    def __str__(self) -> str:
        return "Tensor %r with grad %r" % (self.data, self.grad)
    
    def backward(self, allow_fill=True):

        # print("Backward on", self)

        if self._ctx is None:
            return

        if self.grad is None and allow_fill:
            # fill the first gradient with ones
            assert self.data.size == 1
            self.grad = np.ones_like(self.data)
        
        assert self.grad is not None, "grad is None"

        grads = self._ctx.backward(self._ctx, self.grad)

        if len(self._ctx.parents) == 1:
            grads = [grads]
        
        for t, g in zip(self._ctx.parents, grads):
            if g.shape != t.data.shape:
                print("grad shape must match tensor shape in %r, %r != %r" % (self._ctx, g.shape, t.data.shape))
                assert(False)
            t.grad = g
            t.backward(allow_fill=False)
    
    def mean(self):
        div = Tensor(np.array([1 / self.data.size]))
        return self.sum().mul(div)
        
# The Function is the Context
class Function:
  def __init__(self, *tensors):
    self.parents = tensors
    self.saved_tensors = []

  def save_for_backward(self, *x):
    self.saved_tensors.extend(x)

  # note that due to how partialmethod works, self and arg are switched
  def apply(self, arg, *x):
    ctx = arg(self, *x)
    ret = Tensor(arg.forward(ctx, self.data, *[t.data for t in x]))
    ret._ctx = ctx
    return ret

def register(name, fxn):
  setattr(Tensor, name, partialmethod(fxn.apply, fxn))


# Arithmetic and logic functions 

class Add(Function):
    @staticmethod
    def forward(ctx, x, y):
        return x + y
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output,grad_output
register('add', Add)

class Multiply(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return y * grad_output, x * grad_output
register('mul', Multiply)

class ReLU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return np.maximum(input, 0)
    
    @staticmethod
    def backward(ctx, grad_output):
        input,  = ctx.saved_tensors
        grad_input = grad_output.copy()
        grad_input[input < 0] = 0
        return grad_input
register('relu', ReLU)

class Dot(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x.dot(y)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        grad_x = grad_output.dot(y.T)
        grad_y = grad_output.T.dot(x).T
        return grad_x, grad_y
register('dot', Dot)

class Sum(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return np.array([input.sum()])
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return np.ones_like(input) * grad_output
register('sum', Sum)

class LogSoftmax(Function):
  @staticmethod
  def forward(ctx, input):
    def logsumexp(x):
      #return np.log(np.exp(x).sum(axis=1))
      c = x.max(axis=1)
      return c + np.log(np.exp(x-c.reshape((-1, 1))).sum(axis=1))
    output = input - logsumexp(input).reshape((-1, 1))
    ctx.save_for_backward(output)
    return output

  @staticmethod
  def backward(ctx, grad_output):
    output, = ctx.saved_tensors
    return grad_output - np.exp(output)*grad_output.sum(axis=1).reshape((-1, 1))
register('logsoftmax', LogSoftmax)


class Conv2D(Function):
    @staticmethod
    def forward(ctx, x, w):
        cout, cin ,kh, kw = w.shape
        ret = np.zeros((cout, x.shape[0] - kh + 1, x.shape[1] - kw + 1), dtype=w.dtype)

        for co in range(cout):
            for i in range(ret.shape[1]):
                for j in range(ret.shape[2]):
                    ret[co, i, j] = (x[:, i:i+kh, j:j+kw] * w[co]).sum()
        ctx.save_for_backward(x, w)
        return ret
    
    @staticmethod
    def backward(ctx, grad_output):
        raise Exception("Not implemented")