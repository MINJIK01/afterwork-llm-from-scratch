"""프리트레이닝 (step 10-13). 예: python scripts/pretrain.py --preset m5"""

import argparse
import csv
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
    ap.add_argument("--resume", action="store_true", help="--out 체크포인트에서 모델·optimizer·데이터 순서를 복원해 이어서 학습")
    ap.add_argument("--log", default=None, help="학습 로그 CSV 경로 (기본: --out과 같은 이름의 .csv)")
    args = ap.parse_args()
    cfg_p = PRESETS[args.preset] | {k: v for k, v in vars(args).items() if k in ("max_steps", "batch_size", "lr") and v}
    out = Path(args.out) if args.out else OUT / f"{'draft' if args.preset == 'draft' else 'pretrain'}.pt"
    log_path = Path(args.log) if args.log else out.with_suffix(".csv")

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
    start = 0
    if args.resume and out.exists():
        ck = torch.load(out, map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        gen.set_state(ck["gen"])
        if scaler and ck.get("scale"):
            scaler.scale = ck["scale"]
        start = ck["step"]
        print(f"resumed from {out} at step {start}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    if start and log_path.exists():  # 재개 지점 이후에 기록된 줄은 다시 학습하므로 제거
        with log_path.open() as f:
            rows = list(csv.reader(f))
        with log_path.open("w", newline="") as f:
            csv.writer(f, lineterminator="\n").writerows(rows[:1] + [r for r in rows[1:] if int(r[0]) <= start])
    with open(log_path, "a" if start else "w", newline="") as log_f:
        log = csv.writer(log_f, lineterminator="\n")
        if not start:
            log.writerow(["step", "train_loss", "lr", "grad_norm", "tok_per_s", "val_loss"])
        t0 = time.time()
        for step in range(start, steps):
            lr = cosine_lr(step, cfg_p["lr"], cfg_p["lr"] / 10, cfg_p["warmup"], steps)
            for g in opt.param_groups:
                g["lr"] = lr
            batches = [get_batch(train, bs, T, dev, gen) for _ in range(accum)]
            if amp is None:
                loss, norm = train_step(model, opt, batches)
            else:
                loss, norm = train_step_amp(model, opt, batches, 1.0, amp, scaler)
            if step % 20 == 0:
                tps = bs * T * accum * (step - start + 1) / (time.time() - t0)
                print(f"step {step:5d} | loss {loss:.4f} | lr {lr:.2e} | grad norm {norm:.2f} | {tps:,.0f} tok/s")
                log.writerow([step + 1, f"{loss:.4f}", f"{lr:.3e}", f"{norm:.3f}", round(tps), ""])  # step = 완료한 스텝 수
            if (step + 1) % cfg_p["eval_every"] == 0 or step == steps - 1:
                vl, ppl = evaluate(model, val, bs, T, cfg_p["eval_iters"], dev, torch.Generator().manual_seed(1))
                print(f"[eval] step {step + 1} | val loss {vl:.4f} | ppl {ppl:.2f}")
                log.writerow([step + 1, "", "", "", "", f"{vl:.4f}"])
                log_f.flush()
                save_ckpt(out, model, step=step + 1, val_loss=vl, opt=opt.state_dict(), gen=gen.get_state(),
                          scale=scaler.scale if scaler else None)

    print(f"saved {out} | log {log_path}")
    start = torch.tensor([[tok.special_tokens[EOS]]] * 2, device=dev)
    for row in generate(model, start, T - 1, temperature=0.8, top_k=50, eos_id=tok.special_tokens[EOS]).tolist():
        print("---\n" + tok.decode(row[1:]))


if __name__ == "__main__":
    main()
