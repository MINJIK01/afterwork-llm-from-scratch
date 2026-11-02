"""정렬 결과 비교 (step 15-18, 24). 체크포인트별로 평가 키워드에 대해 샘플링해 다음을 측정한다.

- hit: 키워드 적중률 (규칙 보상의 평균)
- eos: EOS로 끝맺은 비율
- len: 응답 토큰 수 평균 (길이 편향 확인, docs/answers/17)
- KL: 참조 모델(기본 SFT) 대비 시퀀스 KL(π‖ref) 추정 = E_{y~π}[Σ_t log π(y_t)/ref(y_t)]. temperature=1일 때만 정확한 추정
- val: 사전학습 검증 loss (정렬로 인한 언어 모델링 능력 손실 확인)
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from common import DATA, EOS, EVAL_KEYWORDS, OUT, device, load_ckpt, load_tokenizer, response_logprobs, response_mask, rollout
from minillm.trainer import evaluate

DEFAULTS = ["pretrain.pt", "sft.pt", "sft_lora.pt", "dpo.pt", "grpo.pt"]


@torch.no_grad()
def measure(model, ref, tok, dev, val, n, temperature):
    eos = tok.special_tokens[EOS]
    hits, ended, lengths, kls = [], [], [], []
    for kw in EVAL_KEYWORDS:
        p, responses, rewards, out = rollout(model, tok, kw, n, dev, temperature)
        mask = response_mask(out[:, len(p) :], eos)
        kls += ((response_logprobs(model, out, len(p)) - response_logprobs(ref, out, len(p))) * mask).sum(1).tolist()
        hits += rewards
        ended += [eos in r for r in responses]
        lengths += [len(r) for r in responses]
    vl, _ = evaluate(model, val, 16, model.cfg.block_size, 20, dev, torch.Generator().manual_seed(0))
    return [float(np.mean(x)) for x in (hits, ended, lengths, kls)] + [vl]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="*", default=[str(OUT / f) for f in DEFAULTS])
    ap.add_argument("--ref", default=str(OUT / "sft.pt"))
    ap.add_argument("--n", type=int, default=8, help="키워드당 샘플 수")
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    dev, tok = device(), load_tokenizer()
    ref = load_ckpt(args.ref, dev)
    val = np.memmap(DATA / "val.bin", dtype=np.uint16, mode="r")
    if args.temperature != 1.0:
        print("주의: temperature != 1이면 KL 열은 KL(π‖ref)의 비편향 추정이 아니다.")
    print(f"{'checkpoint':16} {'hit':>6} {'eos':>6} {'len':>7} {'KL':>8} {'val':>7}")
    for path in map(Path, args.ckpts):
        if not path.exists():
            continue
        torch.manual_seed(0)
        hit, eos, length, kl, vl = measure(load_ckpt(path, dev), ref, tok, dev, val, args.n, args.temperature)
        print(f"{path.name:16} {hit:6.3f} {eos:6.3f} {length:7.1f} {kl:8.3f} {vl:7.4f}")


if __name__ == "__main__":
    main()
