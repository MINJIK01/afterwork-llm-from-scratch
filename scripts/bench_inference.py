"""KV cache(step 14)와 양자화(step 19) 벤치마크."""

import argparse
import copy

import numpy as np
import torch

from common import DATA, EOS, OUT, Timer, device, load_ckpt, load_tokenizer
from minillm.generate import generate
from minillm.layers import kv_cache_bytes
from minillm.quant import quantize_model, weight_bytes
from minillm.trainer import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(OUT / "pretrain.pt"))
    ap.add_argument("--eval_iters", type=int, default=20)
    args = ap.parse_args()

    dev, tok = device(), load_tokenizer()
    model = load_ckpt(args.ckpt, dev)
    cfg = model.cfg
    hd = cfg.d_model // cfg.n_head

    print("== KV cache ==")
    idx = torch.tensor([[tok.special_tokens[EOS]]], device=dev)
    n = cfg.block_size - 1
    outs = {}
    for use_cache in (False, True):
        with Timer(dev) as t:
            outs[use_cache] = generate(model, idx, n, temperature=0, use_cache=use_cache)
        print(f"use_cache={use_cache!s:5} | {outs[use_cache].shape[1] - 1} tokens | {(outs[use_cache].shape[1] - 1) / t.elapsed:,.0f} tok/s")
    print(f"identical outputs: {torch.equal(outs[False], outs[True])}")
    print(f"KV cache (this model, B=1, T={cfg.block_size}, fp32): {kv_cache_bytes(cfg.n_layer, cfg.n_head, hd, 1, cfg.block_size, 4) / 2**20:.2f} MiB")
    print(f"KV cache (Llama-2-7B, B=8, T=4096, fp16): {kv_cache_bytes(32, 32, 128, 8, 4096, 2) / 2**30:.1f} GiB")

    print("== Weight-only quantization ==")
    val = np.memmap(DATA / "val.bin", dtype=np.uint16, mode="r")
    for name, bits, group in [("fp32", None, None), ("int8/channel", 8, None), ("int4/channel", 4, None), ("int4/g64", 4, 64), ("int4/g32", 4, 32)]:
        m = model if bits is None else quantize_model(copy.deepcopy(model), bits, group)
        loss, ppl = evaluate(m, val, 16, cfg.block_size, args.eval_iters, dev, torch.Generator().manual_seed(0))
        print(f"{name:13} | weights {weight_bytes(m) / 2**20:6.2f} MiB | val loss {loss:.4f} | ppl {ppl:.3f}")


if __name__ == "__main__":
    main()
