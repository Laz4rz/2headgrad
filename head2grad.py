from visualizer import draw_dot

GRAD_ACC_OP = {
    "+": 0,
    "-": 0,
    "*": 1
}


class Value:
    def __init__(self, data, _children=(), _op='', label=""):
        self.data = data
        self.grad = 0 
        self._prev = set(_children)
        self._op = _op
        self.label = label
        
    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __sub__(self, other):
        out = Value(self.data - other.data, (self, other), '-')
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out

    def _backward(self):
        print(self)
        grad_acc_op = GRAD_ACC_OP[self._op]
        for child1 in self._prev:
            grad_acc = 0
            for child2 in self._prev:
                if child1 != child2:
                    grad_acc += child2.data * grad_acc_op
            if grad_acc == 0:
                child1.grad = self.grad
            else:
                child1.grad = self.grad * grad_acc
            if len(child1._prev) > 0:
                child1._backward()

    def backward(self):
        self.grad = 1
        self._backward()

if __name__ == "__main__":
    v1 = Value(1)
    v2 = Value(2)
    v3 = v1 + v2
    v4 = v1 - v2
    v5 = v1 * v2
    print(f"{v1=}, {v2=}, {v3=}, {v4=}, {v5=}")
    draw_dot(v5)
