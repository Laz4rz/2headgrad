import math
from visualizer import draw_dot


class Value:
    def __init__(self, data, _children=(), _op='', label=""):
        self.data = data
        self.grad = 0 
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
        
    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad
        out._backward = _backward

        return out

    def __sub__(self, other):
        out = Value(self.data - other.data, (self, other), '-')

        def _backward():
            self.grad = -1.0 * out.grad
            other.grad = -1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad
        out._backward = _backward

        return out
    
    def tanh(self):
        x = self.data
        tanh = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(tanh, [self], "tanh")

        def _backward():
            self.grad = (1  - tanh**2) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        self.grad = 1
        self._backward()
        for child in self._prev:
            child._backward()
        

if __name__ == "__main__":
    v1 = Value(1)
    v2 = Value(2)
    v3 = v1 + v2
    v4 = v1 - v2
    v5 = v1 * v2
    print(f"{v1=}, {v2=}, {v3=}, {v4=}, {v5=}")
    draw_dot(v5)
