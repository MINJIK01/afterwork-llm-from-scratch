from minillm.align import build_sft_example


def test_masking_and_padding():
    x, y = build_sft_example([10, 11, 12], [20, 21, 2], block_size=8)
    assert x.tolist() == [10, 11, 12, 20, 21, 0, 0, 0]
    assert y.tolist() == [-100, -100, 20, 21, 2, -100, -100, -100]


def test_truncation():
    x, y = build_sft_example([10, 11], [20, 21, 22, 23, 24], block_size=4)
    assert x.tolist() == [10, 11, 20, 21]
    assert y.tolist() == [-100, 20, 21, 22]
