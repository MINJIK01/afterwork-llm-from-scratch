"""GRPO (step 18). 온라인으로 프롬프트당 G개 샘플 → 그룹 상대 advantage → PPO-clip + KL(k3) 업데이트."""

import argparse
import copy
import random
from pathlib import Path

import torch

from common import (
    EOS, OUT, device, keyword_hit_rate, keyword_pool, load_ckpt, load_tokenizer, read_docs,
    response_logprobs, response_mask, rollout, save_ckpt, show_samples,
)
from minillm.align import group_advantages, kl_k3, ppo_clip_loss
from minillm.optim import clip_grad_norm_, configure_optimizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(OUT / "sft.pt"))
    ap.add_argument("--out", default=str(OUT / "grpo.pt"))
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--prompts_per_iter", type=int, default=8)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--inner_epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--kl", type=float, default=0.05)
    ap.add_argument("--eval_every", type=int, default=50)
    args = ap.parse_args()

    dev, tok = device(), load_tokenizer()
    policy = load_ckpt(args.init, dev)
    ref = copy.deepcopy(policy).requires_grad_(False)
    eos, rng = tok.special_tokens[EOS], random.Random(0)
    pool = keyword_pool(read_docs("train"), rng, 2000)
    opt = configure_optimizer(policy, args.lr, weight_decay=0.0)
    print(f"[before] keyword hit rate {keyword_hit_rate(policy, tok, dev):.3f}")

    for it in range(args.iters):
        groups, all_rewards = [], []
        for kw in rng.sample(pool, args.prompts_per_iter):
            p, _, rewards, out = rollout(policy, tok, kw, args.group, dev)
            mask = response_mask(out[:, len(p) :], eos)
            with torch.no_grad():
                old, ref_lp = response_logprobs(policy, out, len(p)), response_logprobs(ref, out, len(p))
            groups.append((out, len(p), mask, old, ref_lp))
            all_rewards += rewards
        adv = group_advantages(torch.tensor(all_rewards, device=dev), args.group).split(args.group)

        for _ in range(args.inner_epochs):
            opt.zero_grad(set_to_none=True)
            for (out, n_p, mask, old, ref_lp), a in zip(groups, adv):
                logp = response_logprobs(policy, out, n_p)
                kl = kl_k3(logp, ref_lp, mask)
                loss = ppo_clip_loss(logp, old, a, mask, args.clip) + args.kl * kl
                (loss / len(groups)).backward()
            clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

        print(f"iter {it:4d} | mean reward {sum(all_rewards) / len(all_rewards):.3f} | kl {kl.item():.4f}")
        if (it + 1) % args.eval_every == 0:
            print(f"[eval] iter {it + 1} | keyword hit rate {keyword_hit_rate(policy, tok, dev):.3f}")

    show_samples(policy, tok, dev)
    save_ckpt(Path(args.out), policy)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
