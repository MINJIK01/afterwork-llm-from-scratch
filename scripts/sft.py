"""SFT (step 15). 문서 속 단어 하나를 키워드로 뽑아 '이 단어로 이야기를 써라' 지시-응답 쌍을 자동 생성한다."""

import argparse
import random
from pathlib import Path

import torch

from common import EOS, OUT, device, keyword_hit_rate, load_ckpt, load_tokenizer, pick_keyword, prompt_text, read_docs, save_ckpt, show_samples
from minillm.align import build_sft_example
from minillm.optim import configure_optimizer, cosine_lr
from minillm.trainer import train_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(OUT / "pretrain.pt"))
    ap.add_argument("--out", default=str(OUT / "sft.pt"))
    ap.add_argument("--n_examples", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    dev, tok = device(), load_tokenizer()
    model = load_ckpt(args.init, dev)
    T, eos, rng = model.cfg.block_size, tok.special_tokens[EOS], random.Random(0)

    xs, ys = [], []
    for doc in read_docs("train"):
        kw = pick_keyword(doc, rng)
        p, r = (tok.encode(prompt_text(kw)), tok.encode(doc) + [eos]) if kw else ([], [])
        if kw and len(p) + len(r) <= T + 1:
            x, y = build_sft_example(p, r, T)
            xs.append(x)
            ys.append(y)
        if len(xs) >= args.n_examples:
            break
    X, Y = torch.stack(xs), torch.stack(ys)
    print(f"SFT examples {len(X):,} | example:\n{tok.decode(X[0][: (Y[0] != -100).nonzero().max() + 2].tolist())}")

    print(f"[before] keyword hit rate {keyword_hit_rate(model, tok, dev):.3f}")
    opt = configure_optimizer(model, args.lr, weight_decay=0.0)
    gen = torch.Generator().manual_seed(0)
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, args.lr, args.lr / 10, 50, args.steps)
        i = torch.randint(len(X), (args.batch_size,), generator=gen)
        loss, _ = train_step(model, opt, [(X[i].to(dev), Y[i].to(dev))])
        if step % 50 == 0:
            print(f"step {step:4d} | sft loss {loss:.4f}")

    print(f"[after] keyword hit rate {keyword_hit_rate(model, tok, dev):.3f}")
    show_samples(model, tok, dev)
    save_ckpt(Path(args.out), model)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
