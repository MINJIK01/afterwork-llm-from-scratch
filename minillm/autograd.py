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
            # BEGIN TODO (step 01)
            raise NotImplementedError("TODO: step 01 (docs/01_*.md)")
            # END TODO

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # BEGIN TODO (step 01)
            raise NotImplementedError("TODO: step 01 (docs/01_*.md)")
            # END TODO

        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "지수는 상수만 지원"
        out = Value(self.data**k, (self,), f"**{k}")

        def _backward():
            # BEGIN TODO (step 01)
            raise NotImplementedError("TODO: step 01 (docs/01_*.md)")
            # END TODO

        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            # BEGIN TODO (step 01)
            raise NotImplementedError("TODO: step 01 (docs/01_*.md)")
            # END TODO

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            # BEGIN TODO (step 01)
            raise NotImplementedError("TODO: step 01 (docs/01_*.md)")
            # END TODO

        out._backward = _backward
        return out

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), "tanh")

        def _backward():
            # BEGIN TODO (step 01)
            raise NotImplementedError("TODO: step 01 (docs/01_*.md)")
            # END TODO

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            # BEGIN TODO (step 01)
            raise NotImplementedError("TODO: step 01 (docs/01_*.md)")
            # END TODO

        out._backward = _backward
        return out

    def backward(self):
        """self를 루트로 위상 정렬 역순으로 _backward 호출. 루트의 grad는 1."""
        # BEGIN TODO (step 01)
        raise NotImplementedError("TODO: step 01 (docs/01_*.md)")
        # END TODO

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
