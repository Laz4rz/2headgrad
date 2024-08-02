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
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    # def __sub__(self, other):
    #     other = other if isinstance(Value) else Value(other)
    #     out = Value(self.data - other.data, (self, other), '-')

    #     def _backward():
    #         self.grad += -1.0 * out.grad
    #         other.grad += -1.0 * out.grad
    #     out._backward = _backward

    #     return out

    # sub can also be wrote using previously defined operations
    def __sub__(self, other):
        return self + (-other)
    
    def __neg__(self):
        return self * (-1)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __truediv__(self, other):
        return self * other ** (-1)
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int/float"
        out = Value(self.data ** other, (self,), f"pow(2)")

        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other 
    
    def __rsub__(self, other):
        return self - other 
    
    def __rmul__(self, other):
        return self * other 
    
    def tanh(self):
        x = self.data
        val = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(val, [self], "tanh")

        def _backward():
            self.grad = (1  - val**2) * out.grad # 1 - tanh(x)^2
        out._backward = _backward

        return out
    
    def exp(self):
        val = math.exp(self.data)
        out = Value(val, [self], "exp")

        def _backward():
            self.grad += val * out.grad # e^x
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
    

# i guess this is correct? and I like it cause I wrote it? **i GUESS correct**
# UPDATE: its wrong
def build_topological(val, topological):
    if val._prev is None:
        topological.append(val)
    else:
        for child in val._prev:
            build_topological(child, topological)
        topological.append(val)        
        return list(reversed(topological))
        

if __name__ == "__main__":
    v1 = Value(1)
    v2 = Value(2)
    v3 = v1 + v2
    v4 = v1 - v2
    v5 = v1 * v2
    print(f"{v1=}, {v2=}, {v3=}, {v4=}, {v5=}")
    draw_dot(v5)
