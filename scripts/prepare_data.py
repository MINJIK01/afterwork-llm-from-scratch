"""말뭉치 다운로드 → BPE 학습(step 05) → 토큰화 → data/{train,val}.bin (step 11)."""

import argparse
import random
import time
import urllib.request

import numpy as np

from common import DATA, EOS, SPECIALS
from minillm.data import pack_documents
from minillm.tokenizer import BPETokenizer

HF = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
URLS = {
    "tinystories_train": f"{HF}/TinyStoriesV2-GPT4-train.txt",
    "tinystories_val": f"{HF}/TinyStoriesV2-GPT4-valid.txt",
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
}


def download(url, max_mb=None):
    print(f"downloading {url} (max {max_mb} MB)")
    with urllib.request.urlopen(url) as r:
        raw = r.read(int(max_mb * 1e6)) if max_mb else r.read()
    return raw.decode("utf-8", errors="ignore")


def load_docs(args):
    if args.dataset == "tinystories":
        train = download(URLS["tinystories_train"], args.train_mb).split(EOS)[:-1]  # 마지막은 잘린 문서
        val = download(URLS["tinystories_val"], args.val_mb).split(EOS)[:-1]
        return [d.strip() for d in train if d.strip()], [d.strip() for d in val if d.strip()]
    paras, docs, cur = download(URLS["shakespeare"]).split("\n\n"), [], ""
    for p in paras:
        cur = f"{cur}\n\n{p}" if cur else p
        if len(cur) >= 250:
            docs.append(cur)
            cur = ""
    random.Random(0).shuffle(docs)
    n_val = len(docs) // 10
    return docs[n_val:], docs[:n_val]


def encode_docs(tok, docs, eos_id, chunk=5000):
    return np.concatenate([pack_documents([tok.encode(d) for d in docs[i : i + chunk]], eos_id) for i in range(0, len(docs), chunk)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["tinystories", "shakespeare"], default="tinystories")
    ap.add_argument("--train_mb", type=float, default=200)
    ap.add_argument("--val_mb", type=float, default=5)
    ap.add_argument("--vocab_size", type=int, default=4096)
    ap.add_argument("--bpe_train_mb", type=float, default=2, help="BPE 학습에 쓸 텍스트 크기 (순수 Python이라 느림)")
    args = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    train_docs, val_docs = load_docs(args)
    print(f"docs: train {len(train_docs):,}, val {len(val_docs):,}")

    t = time.time()
    tok = BPETokenizer.train("\n".join(train_docs)[: int(args.bpe_train_mb * 1e6)], args.vocab_size, SPECIALS)
    tok.save(DATA / "tokenizer.json")
    print(f"BPE trained: vocab {tok.vocab_size} ({time.time() - t:.0f}s)")

    eos_id = tok.special_tokens[EOS]
    for split, docs in [("train", train_docs), ("val", val_docs)]:
        t = time.time()
        (DATA / f"{split}.txt").write_text(f"\n{EOS}\n".join(docs))
        arr = encode_docs(tok, docs, eos_id)
        arr.tofile(DATA / f"{split}.bin")
        n_bytes = sum(len(d.encode()) for d in docs)
        print(f"{split}: {len(arr):,} tokens, {n_bytes / len(arr):.2f} bytes/token ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
