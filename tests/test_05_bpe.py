from collections import Counter

from minillm.tokenizer import BPETokenizer, get_pair_counts, merge

SPECIALS = ("<|endoftext|>", "<|user|>")


def test_get_pair_counts_and_merge():
    words = {(1, 2, 3): 2, (2, 3): 1}
    assert get_pair_counts(words) == Counter({(1, 2): 2, (2, 3): 3})
    assert merge((1, 1, 1, 2, 1, 1), (1, 1), 9) == (9, 1, 2, 9)


def test_train_wikipedia_example():
    tok = BPETokenizer.train("aaabdaaabac", 256 + 3)
    assert list(tok.merges.items()) == [((97, 97), 256), ((97, 98), 257), ((256, 257), 258)]
    assert tok.vocab[258] == b"aaab"


def test_roundtrip_and_specials(tmp_path):
    text = "Once upon a time, Lily saw a big dog. The dog was happy!\n" * 20
    tok = BPETokenizer.train(text, 290, special_tokens=SPECIALS)
    assert tok.vocab_size == 290
    assert sorted(tok.special_tokens.values()) == [288, 289]

    sample = "Lily안녕 🙂 dog's  tab\there<|endoftext|><|user|>x"
    ids = tok.encode(sample)
    assert tok.decode(ids) == sample
    assert tok.special_tokens["<|endoftext|>"] in ids
    assert tok.encode("<|endoftext|>") == [tok.special_tokens["<|endoftext|>"]]

    assert len(tok.encode(text)) < len(text.encode()) / 2

    tok.save(tmp_path / "tok.json")
    tok2 = BPETokenizer.load(tmp_path / "tok.json")
    assert tok2.encode(sample) == ids
