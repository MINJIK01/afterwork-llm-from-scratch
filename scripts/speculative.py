"""Speculative decoding (step 20). 타깃(pretrain.pt) + 드래프트(draft.pt)."""

import argparse

import torch

from common import EOS, OUT, Timer, device, load_ckpt, load_tokenizer
from minillm.generate import generate, speculative_generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=str(OUT / "pretrain.pt"))
    ap.add_argument("--draft", default=str(OUT / "draft.pt"))
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    dev, tok = device(), load_tokenizer()
    target, draft = load_ckpt(args.target, dev), load_ckpt(args.draft, dev)
    idx = torch.tensor([[tok.special_tokens[EOS]] + tok.encode("Once upon a time")], device=dev)
    n = min(args.n, target.cfg.block_size - idx.shape[1])

    for temp in (0.0, 1.0):
        torch.manual_seed(0)
        with Timer(dev) as t_ar:
            ref = generate(target, idx, n, temperature=temp)
        with Timer(dev) as t_sp:
            out, accepted = speculative_generate(target, draft, idx, n, args.gamma, temp)
        n_ar, n_sp = ref.shape[1] - idx.shape[1], out.shape[1] - idx.shape[1]
        print(f"== temperature {temp} ==")
        print(f"autoregressive : {n_ar / t_ar.elapsed:7,.0f} tok/s ({n_ar} target forwards)")
        print(f"speculative γ={args.gamma}: {n_sp / t_sp.elapsed:7,.0f} tok/s ({len(accepted)} target forwards, "
              f"평균 {sum(accepted) / len(accepted):.2f} tokens/round)")
        if temp == 0:
            print(f"greedy 출력 일치: {torch.equal(out, ref[:, : out.shape[1]])}")
        print(tok.decode(out[0, 1:].tolist()))


if __name__ == "__main__":
    main()
