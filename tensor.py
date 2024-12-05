from functools import partialmethod
import numpy as np


class Context:
    
    def __init__(self, arg, *tensors):
        self.arg = arg
        self.parents = tensors
        self.saved_tensors = []

    def saved_for_bacward(self, *x):
        self.saved_tensors.extend(x)

class Tensor:

    def __init__(self, data):

        if (type(data) is not np.ndarray):
            raise ValueError("Data must be of type numpy.ndarray", data)
            assert False
        self.data = data
        self.grad = None

        # dont understand the use of this part
        self._ctx = None
    def __str__(self) -> str:
        return "Tensor %r with grad %r" % (self.data, self.grad)
    
    def backward(self, allow_fill=True):

        print("Backward on", self)

        if self._ctx is None:
            return

        if self.grad is None and allow_fill:
            # fill the first gradient with ones
            assert self.data.size == 1
            self.grad = np.ones_like(self.data)
        
        assert self.grad is not None, "grad is None"

        grads = self._ctx.arg.backward(self._ctx, self.grad)

        if len(self._ctx.parents) == 1:
            grads = [grads]
        
        for t, grad in zip(self._ctx.parents, grads):
            if grad.shape != t.data.shape:
                print("grad shape must match tensor shape in %r, %r != %r" % (self._ctx.arg, grad.shape, t.data.shape))
                assert(False)
            t.grad = grad
            t.backward(allow_fill=False)
    
    def mean(self):
        div = Tensor(np.array([1 / self.data.size]))
        return self.sum().mul(div)
        
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

class Mulitply(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.saved_for_bacward(a, b)
        return a * b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return grad_output * b, grad_output * a

class Dot(Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.saved_for_bacward(input, weight)
        return input.dot(weight)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        return grad_output.dot(weight.T), input.Y.dot(grad_output)