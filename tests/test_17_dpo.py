import math

import torch
import torch.nn.functional as F

from minillm.align import dpo_loss, sequence_logprobs, token_logprobs


def test_token_and_sequence_logprobs(tiny_model):
    idx = torch.randint(0, 64, (2, 8))
    labels = torch.randint(0, 64, (2, 8))
    labels[0, :3] = -100
    logits, _ = tiny_model(idx)
    ref_tok = F.log_softmax(logits, -1).gather(-1, labels.clamp(min=0)[..., None])[..., 0]
    assert torch.allclose(token_logprobs(logits, labels)[0, 3:], ref_tok[0, 3:], atol=1e-5)
    ref_seq = (ref_tok * (labels != -100)).sum(-1)
    assert torch.allclose(sequence_logprobs(tiny_model, idx, labels), ref_seq, atol=1e-4)


def test_dpo_loss():
    pc, pr = torch.tensor([-5.0, -3.0]), torch.tensor([-6.0, -2.0])
    rc, rr = torch.tensor([-5.5, -3.5]), torch.tensor([-5.5, -2.5])
    loss, cr, rj = dpo_loss(pc, pr, rc, rr, beta=0.5)
    z = 0.5 * ((pc - rc) - (pr - rr))
    assert torch.allclose(loss, -F.logsigmoid(z).mean())
    assert torch.allclose(cr, 0.5 * (pc - rc)) and torch.allclose(rj, 0.5 * (pr - rr))
    assert not cr.requires_grad


def test_dpo_at_reference_is_log2_and_gradient_direction():
    pc = torch.tensor([-4.0], requires_grad=True)
    pr = torch.tensor([-4.0], requires_grad=True)
    loss, _, _ = dpo_loss(pc, pr, pc.detach(), pr.detach(), beta=0.1)
    assert math.isclose(loss.item(), math.log(2), rel_tol=1e-6)
    loss.backward()
    assert pc.grad.item() < 0 < pr.grad.item()
