import json
import re
from collections import Counter

# GPT-2 pre-tokenization 정규식의 ASCII 근사. 모든 문자를 빠짐없이 덮는다.
PAT = re.compile(r"""'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""")


def get_pair_counts(words):
    """words: {토큰 id 튜플: 빈도}. 인접 pair별 (빈도 가중) 등장 횟수 Counter. (step 05)"""
    # BEGIN TODO (step 05)
    raise NotImplementedError("TODO: step 05 (docs/05_*.md)")
    # END TODO


def merge(ids, pair, new_id):
    """ids에서 pair가 연속으로 나오는 곳을 왼쪽부터 겹치지 않게 new_id로 치환한 튜플. (step 05)"""
    # BEGIN TODO (step 05)
    raise NotImplementedError("TODO: step 05 (docs/05_*.md)")
    # END TODO


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
        # BEGIN TODO (step 05)
        raise NotImplementedError("TODO: step 05 (docs/05_*.md)")
        # END TODO
        n = 256 + len(merges)
        return cls(merges, {s: n + j for j, s in enumerate(special_tokens)})

    def _encode_chunk(self, chunk):
        """bytes → ids. 학습 때 먼저 만들어진 merge(작은 id)부터 적용. (step 05)"""
        # BEGIN TODO (step 05)
        raise NotImplementedError("TODO: step 05 (docs/05_*.md)")
        # END TODO

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
        # BEGIN TODO (step 05)
        raise NotImplementedError("TODO: step 05 (docs/05_*.md)")
        # END TODO

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"merges": [[a, b, i] for (a, b), i in self.merges.items()], "special_tokens": self.special_tokens}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls({(a, b): i for a, b, i in d["merges"]}, d["special_tokens"])
