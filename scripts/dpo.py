"""DPO (step 16-17). SFT 모델로 프롬프트당 k개 샘플 → 규칙 기반 보상으로 (chosen, rejected) 쌍 구성 → DPO 학습."""

import argparse
import copy
import random
from pathlib import Path

import torch

from common import OUT, device, keyword_hit_rate, keyword_pool, load_ckpt, load_tokenizer, read_docs, rollout, save_ckpt, show_samples
from minillm.align import build_sft_example, dpo_loss, sequence_logprobs
from minillm.optim import clip_grad_norm_, configure_optimizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(OUT / "sft.pt"))
    ap.add_argument("--out", default=str(OUT / "dpo.pt"))
    ap.add_argument("--n_prompts", type=int, default=1000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    args = ap.parse_args()

    dev, tok = device(), load_tokenizer()
    policy = load_ckpt(args.init, dev)
    ref = copy.deepcopy(policy).requires_grad_(False)
    T, rng = policy.cfg.block_size, random.Random(0)
    print(f"[before] keyword hit rate {keyword_hit_rate(policy, tok, dev):.3f}")

    pairs = []
    for i, kw in enumerate(rng.choices(keyword_pool(read_docs("train"), rng, args.n_prompts), k=args.n_prompts)):
        p, responses, rewards, _ = rollout(policy, tok, kw, args.k, dev)
        pos = [r for r, w in zip(responses, rewards) if w > 0]
        neg = [r for r, w in zip(responses, rewards) if w == 0]
        if pos and neg:
            pairs.append(build_sft_example(p, rng.choice(pos), T) + build_sft_example(p, rng.choice(neg), T))
        if i % 100 == 0:
            print(f"rollout {i}/{args.n_prompts} | pairs {len(pairs)}")
    if not pairs:
        raise SystemExit("선호쌍이 없습니다: SFT 모델의 보상이 전부 0 또는 전부 1입니다. SFT 적중률을 먼저 확인하세요.")
    xc, yc, xr, yr = (torch.stack(t).to(dev) for t in zip(*pairs))
    print(f"preference pairs {len(xc)}")

    with torch.no_grad():
        ref_c = torch.cat([sequence_logprobs(ref, xc[i : i + 64], yc[i : i + 64]) for i in range(0, len(xc), 64)])
        ref_r = torch.cat([sequence_logprobs(ref, xr[i : i + 64], yr[i : i + 64]) for i in range(0, len(xr), 64)])

    opt = configure_optimizer(policy, args.lr, weight_decay=0.0)
    step = 0
    for epoch in range(args.epochs):
        for b in torch.randperm(len(xc)).split(args.batch_size):
            loss, rc, rr = dpo_loss(
                sequence_logprobs(policy, xc[b], yc[b]), sequence_logprobs(policy, xr[b], yr[b]), ref_c[b], ref_r[b], args.beta
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            if step % 20 == 0:
                acc = (rc > rr).float().mean().item()
                print(f"epoch {epoch} step {step:4d} | dpo loss {loss.item():.4f} | reward acc {acc:.2f} | margin {(rc - rr).mean().item():.3f}")
            step += 1

    print(f"[after] keyword hit rate {keyword_hit_rate(policy, tok, dev):.3f}")
    show_samples(policy, tok, dev)
    save_ckpt(Path(args.out), policy)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
