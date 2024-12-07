

class SGD:
    def __init__(self, params, lr = 0.001):
        self.params = params
        self.lr = lr
    
    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad