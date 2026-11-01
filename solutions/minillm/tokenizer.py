import json
import re
from collections import Counter

# GPT-2 pre-tokenization 정규식의 ASCII 근사. 모든 문자를 빠짐없이 덮는다.
PAT = re.compile(r"""'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""")


def get_pair_counts(words):
    """words: {토큰 id 튜플: 빈도}. 인접 pair별 (빈도 가중) 등장 횟수 Counter. (step 05)"""
    # BEGIN SOLUTION (step 05)
    counts = Counter()
    for w, c in words.items():
        for pair in zip(w, w[1:]):
            counts[pair] += c
    return counts
    # END SOLUTION


def merge(ids, pair, new_id):
    """ids에서 pair가 연속으로 나오는 곳을 왼쪽부터 겹치지 않게 new_id로 치환한 튜플. (step 05)"""
    # BEGIN SOLUTION (step 05)
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)
    # END SOLUTION


class BPETokenizer:
    """Byte-level BPE (Sennrich et al. 2016, GPT-2). (step 05)"""

    def __init__(self, merges=None, special_tokens=None):
        self.merges = dict(merges or {})  # (a, b) -> new_id, 삽입 순서 = merge 우선순위
        self.special_tokens = dict(special_tokens or {})  # str -> id
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), i in self.merges.items():
            self.vocab[i] = self.vocab[a] + self.vocab[b]
        for s, i in self.special_tokens.items():
            self.vocab[i] = s.encode()
        self._special_re = (
            re.compile("(" + "|".join(map(re.escape, self.special_tokens)) + ")") if self.special_tokens else None
        )
        self._cache = {}

    @property
    def vocab_size(self):
        return len(self.vocab)

    @classmethod
    def train(cls, text, vocab_size, special_tokens=()):
        """빈도 최대 pair를 반복 병합 (동률이면 pair 튜플이 사전순으로 작은 것). 새 id는 256부터 순서대로."""
        for s in special_tokens:
            text = text.replace(s, " ")
        num_merges = vocab_size - 256 - len(special_tokens)
        merges = {}
        # BEGIN SOLUTION (step 05)
        words = Counter(tuple(w.encode()) for w in PAT.findall(text))
        for i in range(num_merges):
            counts = get_pair_counts(words)
            if not counts:
                break
            pair = max(counts, key=lambda p: (counts[p], -p[0], -p[1]))
            merges[pair] = 256 + i
            words = {merge(w, pair, 256 + i): c for w, c in words.items()}
        # END SOLUTION
        n = 256 + len(merges)
        return cls(merges, {s: n + j for j, s in enumerate(special_tokens)})

    def _encode_chunk(self, chunk):
        """bytes → ids. 학습 때 먼저 만들어진 merge(작은 id)부터 적용. (step 05)"""
        # BEGIN SOLUTION (step 05)
        ids = tuple(chunk)
        while len(ids) >= 2:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return list(ids)
        # END SOLUTION

    def encode(self, text):
        out = []
        for part in self._special_re.split(text) if self._special_re else [text]:
            if part in self.special_tokens:
                out.append(self.special_tokens[part])
                continue
            for w in PAT.findall(part):
                if w not in self._cache:
                    self._cache[w] = self._encode_chunk(w.encode())
                out.extend(self._cache[w])
        return out

    def decode(self, ids):
        """ids → str. 불완전한 UTF-8 시퀀스는 replacement 문자로. (step 05)"""
        # BEGIN SOLUTION (step 05)
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
        # END SOLUTION

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"merges": [[a, b, i] for (a, b), i in self.merges.items()], "special_tokens": self.special_tokens}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls({(a, b): i for a, b, i in d["merges"]}, d["special_tokens"])
