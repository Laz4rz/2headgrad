from visualizer import draw_dot


class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0 
        self._prev = set(_children)
        self._op = _op

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


if __name__ == "__main__":
    v1 = Value(1)
    v2 = Value(2)
    v3 = v1 + v2
    v4 = v1 - v2
    v5 = v1 * v2
    print(f"{v1=}, {v2=}, {v3=}, {v4=}, {v5=}")
    draw_dot(v5)
