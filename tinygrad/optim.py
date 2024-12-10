from tinygrad.tensor import Tensor
import numpy as np

class SGD:
    def __init__(self, params, lr = 0.001):
        self.params = params
        self.lr = lr
    
    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad

class Adam:
    def __init__(self, params, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps # Correction factor to address / 0
        self.t = 0
        self.m = [Tensor(np.zeros_like(p.data)) for p in params]
        self.v = [Tensor(np.zeros_like(p.data)) for p in params]
        
    def step(self):
        # This tracks the steps of optimization, which are equivalent to the number of times you step().
        self.t += 1
        # Only reason for looping is to address multiple params, which can be of diff. shapes.
        for i, p in enumerate(self.params):
            # Calculating first and second momentum.
            self.m[i].data = self.beta1 * self.m[i].data + (1 - self.beta1) * p.grad
            self.v[i].data = self.beta2 * self.v[i].data + (1 - self.beta2) * p.grad * p.grad
            mhat = self.m[i].data / (1 - self.beta1**self.t)
            vhat = self.v[i].data / (1 - self.beta2**self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)