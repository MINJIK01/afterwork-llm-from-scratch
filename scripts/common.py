import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solutions" if os.environ.get("MINILLM_SOLUTIONS") == "1" else ROOT))

import torch  # noqa: E402

from minillm.generate import generate  # noqa: E402
from minillm.model import GPT, GPTConfig  # noqa: E402
from minillm.tokenizer import BPETokenizer  # noqa: E402

DATA, OUT = ROOT / "data", ROOT / "out"
EOS = "<|endoftext|>"
SPECIALS = (EOS, "<|user|>", "<|assistant|>")

# 정렬 단계 평가용 키워드 (학습용 키워드 풀에서는 제외)
EVAL_KEYWORDS = [
    "ball", "tree", "dog", "cat", "flower", "boat", "cake", "bird", "garden", "hat",
    "box", "park", "rain", "toy", "star", "moon", "apple", "book", "fish", "car",
]
STOPWORDS = set(
    "that this with they them their there then than were have from what when said very were your just like into "
    "been will would could should about after before because while where which some much many more most only also "
    "again other these those every each once upon time thing things".split()
)


def device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()


class Timer:
    def __init__(self, dev):
        self.dev = dev

    def __enter__(self):
        sync(self.dev)
        self.t = time.perf_counter()
        return self

    def __exit__(self, *_):
        sync(self.dev)
        self.elapsed = time.perf_counter() - self.t


def save_ckpt(path, model, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": vars(model.cfg), "model": model.state_dict(), **extra}, path)


def load_ckpt(path, dev):
    ck = torch.load(path, map_location=dev)
    model = GPT(GPTConfig(**ck["cfg"])).to(dev)
    model.load_state_dict(ck["model"])
    return model


def load_tokenizer():
    return BPETokenizer.load(DATA / "tokenizer.json")


def read_docs(split):
    return [d.strip() for d in (DATA / f"{split}.txt").read_text().split(EOS) if d.strip()]


def prompt_text(keyword):
    return f'<|user|>Write a story that uses the word "{keyword}".<|assistant|>'


def pick_keyword(doc, rng):
    words = [w.lower() for w in re.findall(r"[A-Za-z]{4,}", doc) if w.lower() not in STOPWORDS]
    return rng.choice(words) if words else None


def keyword_pool(docs, rng, n):
    pool = {pick_keyword(d, rng) for d in docs[: n * 5]} - set(EVAL_KEYWORDS) - {None}
    return sorted(pool)


def reward(text, ended, keyword):
    """완결(EOS 생성)되었고 키워드를 단어 단위로 포함하면 1, 아니면 0."""
    return float(ended and re.search(rf"\b{re.escape(keyword)}\b", text, re.IGNORECASE) is not None)


def split_response(ids, eos_id):
    """응답 토큰을 첫 EOS까지 자르고 (응답 토큰(EOS 포함), 완결 여부) 반환."""
    return (ids[: ids.index(eos_id) + 1], True) if eos_id in ids else (ids, False)


@torch.no_grad()
def rollout(model, tok, keyword, n, dev, temperature=1.0):
    """같은 프롬프트로 n개 샘플링. (프롬프트 ids, 응답 ids 리스트, 보상 리스트, 원본 출력 텐서) 반환."""
    eos = tok.special_tokens[EOS]
    p = tok.encode(prompt_text(keyword))
    out = generate(model, torch.tensor([p] * n, device=dev), model.cfg.block_size - len(p), temperature, eos_id=eos)
    responses, rewards = [], []
    for row in out[:, len(p) :].tolist():
        resp, ended = split_response(row, eos)
        responses.append(resp)
        rewards.append(reward(tok.decode([t for t in resp if t != eos]), ended, keyword))
    return p, responses, rewards, out


def keyword_hit_rate(model, tok, dev, keywords=EVAL_KEYWORDS, n=4, temperature=0.8):
    rewards = [r for kw in keywords for r in rollout(model, tok, kw, n, dev, temperature)[2]]
    return sum(rewards) / len(rewards)


def show_samples(model, tok, dev, keywords=("dog", "cake"), temperature=0.8):
    for kw in keywords:
        p, responses, rewards, _ = rollout(model, tok, kw, 1, dev, temperature)
        print(f"--- [{kw}] reward={rewards[0]}\n{tok.decode(responses[0])}")
