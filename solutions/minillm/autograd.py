import math


class Value:
    """스칼라 reverse-mode 자동미분 노드. (step 01)"""

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # BEGIN SOLUTION (step 01)
            self.grad += out.grad
            other.grad += out.grad
            # END SOLUTION

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # BEGIN SOLUTION (step 01)
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            # END SOLUTION

        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "지수는 상수만 지원"
        out = Value(self.data**k, (self,), f"**{k}")

        def _backward():
            # BEGIN SOLUTION (step 01)
            self.grad += k * self.data ** (k - 1) * out.grad
            # END SOLUTION

        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            # BEGIN SOLUTION (step 01)
            self.grad += out.data * out.grad
            # END SOLUTION

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            # BEGIN SOLUTION (step 01)
            self.grad += out.grad / self.data
            # END SOLUTION

        out._backward = _backward
        return out

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), "tanh")

        def _backward():
            # BEGIN SOLUTION (step 01)
            self.grad += (1 - out.data**2) * out.grad
            # END SOLUTION

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            # BEGIN SOLUTION (step 01)
            self.grad += (self.data > 0) * out.grad
            # END SOLUTION

        out._backward = _backward
        return out

    def backward(self):
        """self를 루트로 위상 정렬 역순으로 _backward 호출. 루트의 grad는 1."""
        # BEGIN SOLUTION (step 01)
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._prev:
                    build(c)
                topo.append(v)

        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
        # END SOLUTION

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other if isinstance(other, Value) else Value(other)) ** -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Value(other) - self

    def __rtruediv__(self, other):
        return Value(other) / self

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
