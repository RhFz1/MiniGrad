from functools import partialmethod
import numpy as np


class Context:
    
    def __init__(self):
        self.saved_tensors = []

    def saved_for_bacward(self, *x):
        self.saved_tensors.extend(x)

class Tensor:

    def __init__(self, data, _children=()):
        self.data = data
        self.grad = np.zeros_like(self.data)

        # dont understand the use of this part
        self._prev = set(_children)

class Function:
    def apply(self, arg, *x):
        ctx = Context()
        x = [self] + list(x)
        ret = Tensor(arg.forward(ctx, *[t.data for t in x]))
        return ret

def register(name, fn):
    setattr(Tensor, name, partialmethod(fn.apply, fn))


class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.saved_for_bacward(a, b)
        return a + b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return grad_output, grad_output