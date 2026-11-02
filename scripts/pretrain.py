"""프리트레이닝 (step 10-13). 예: python scripts/pretrain.py --preset m5"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from common import DATA, EOS, OUT, device, load_tokenizer, save_ckpt
from minillm.data import get_batch
from minillm.generate import generate
from minillm.model import GPT, GPTConfig
from minillm.optim import configure_optimizer, cosine_lr
from minillm.precision import LossScaler, enable_checkpointing, train_step_amp
from minillm.trainer import evaluate, train_step

PRESETS = {
    # CPU에서 수 분 내 끝나는 동작 확인용
    "tiny": dict(model=dict(n_layer=2, n_head=2, d_model=64, ffn_hidden=192, block_size=192),
                 batch_size=16, grad_accum=1, max_steps=300, lr=2e-3, warmup=30, eval_every=100, eval_iters=10),
    # M5 Mac (MPS) 기준 ~12M 파라미터, 약 1~2시간
    "m5": dict(model=dict(n_layer=6, n_head=6, d_model=384, ffn_hidden=1024, block_size=256),
               batch_size=32, grad_accum=2, max_steps=4000, lr=1e-3, warmup=200, eval_every=500, eval_iters=40),
    # speculative decoding(step 20)용 드래프트 모델
    "draft": dict(model=dict(n_layer=2, n_head=4, d_model=128, ffn_hidden=384, block_size=256),
                  batch_size=32, grad_accum=1, max_steps=3000, lr=2e-3, warmup=100, eval_every=500, eval_iters=20),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="m5")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_steps", type=int)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_kv_head", type=int, help="GQA KV head 수 (step 22). 생략하면 MHA")
    ap.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32", help="mixed precision (step 23)")
    ap.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing (step 23)")
    args = ap.parse_args()
    cfg_p = PRESETS[args.preset] | {k: v for k, v in vars(args).items() if k in ("max_steps", "batch_size", "lr") and v}
    out = Path(args.out) if args.out else OUT / f"{'draft' if args.preset == 'draft' else 'pretrain'}.pt"

    torch.manual_seed(args.seed)
    dev, tok = device(), load_tokenizer()
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, n_kv_head=args.n_kv_head, **cfg_p["model"])).to(dev)
    if args.grad_ckpt:
        enable_checkpointing(model)
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.dtype)
    scaler = LossScaler() if args.dtype == "fp16" else None
    print(f"device {dev} | params {model.num_params() / 1e6:.2f}M | preset {args.preset} | dtype {args.dtype}"
          f" | n_kv_head {args.n_kv_head or cfg_p['model']['n_head']} | grad_ckpt {args.grad_ckpt}")

    train = np.memmap(DATA / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(DATA / "val.bin", dtype=np.uint16, mode="r")
    bs, T, accum, steps = cfg_p["batch_size"], cfg_p["model"]["block_size"], cfg_p["grad_accum"], cfg_p["max_steps"]
    print(f"train tokens {len(train):,} | tokens/step {bs * T * accum:,} | total {bs * T * accum * steps / 1e6:.1f}M")

    opt = configure_optimizer(model, cfg_p["lr"], weight_decay=0.1)
    gen = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    for step in range(steps):
        lr = cosine_lr(step, cfg_p["lr"], cfg_p["lr"] / 10, cfg_p["warmup"], steps)
        for g in opt.param_groups:
            g["lr"] = lr
        batches = [get_batch(train, bs, T, dev, gen) for _ in range(accum)]
        if amp is None:
            loss, norm = train_step(model, opt, batches)
        else:
            loss, norm = train_step_amp(model, opt, batches, 1.0, amp, scaler)
        if step % 20 == 0:
            tps = bs * T * accum * (step + 1) / (time.time() - t0)
            print(f"step {step:5d} | loss {loss:.4f} | lr {lr:.2e} | grad norm {norm:.2f} | {tps:,.0f} tok/s")
        if (step + 1) % cfg_p["eval_every"] == 0 or step == steps - 1:
            vl, ppl = evaluate(model, val, bs, T, cfg_p["eval_iters"], dev, torch.Generator().manual_seed(1))
            print(f"[eval] step {step + 1} | val loss {vl:.4f} | ppl {ppl:.2f}")
            save_ckpt(out, model, step=step + 1, val_loss=vl)

    print(f"saved {out}")
    start = torch.tensor([[tok.special_tokens[EOS]]] * 2, device=dev)
    for row in generate(model, start, T - 1, temperature=0.8, top_k=50, eos_id=tok.special_tokens[EOS]).tolist():
        print("---\n" + tok.decode(row[1:]))


if __name__ == "__main__":
    main()
