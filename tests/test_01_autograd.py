import math

import torch

from minillm.autograd import Value


def test_expression_matches_torch():
    a, b = Value(-2.0), Value(3.0)
    c = a * b + b**2
    d = (c / a).tanh() + (b.exp() - a).log() + (a * -1).relu() - c.relu()
    d.backward()

    at = torch.tensor(-2.0, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    ct = at * bt + bt**2
    dt = (ct / at).tanh() + (bt.exp() - at).log() + (at * -1).relu() - ct.relu()
    dt.backward()

    assert math.isclose(d.data, dt.item(), rel_tol=1e-9)
    assert math.isclose(a.grad, at.grad.item(), rel_tol=1e-9)
    assert math.isclose(b.grad, bt.grad.item(), rel_tol=1e-9)


def test_reused_node_accumulates_grad():
    a = Value(3.0)
    b = a * a + a  # db/da = 2a + 1
    b.backward()
    assert a.grad == 7.0


def test_deep_chain_topological_order():
    x = Value(0.5)
    y = x
    for _ in range(50):
        y = y * 1.01 + 0.0
    y.backward()
    assert math.isclose(x.grad, 1.01**50, rel_tol=1e-9)
