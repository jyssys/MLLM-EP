import numpy as np

from frontier_delta.emulator import decode_word_bitmap, encode_word_bitmap


def test_word_bitmap_round_trip():
    rng = np.random.default_rng(20260918)
    predictor = rng.integers(0, 65536, size=2048, dtype=np.uint16)
    current = predictor.copy()
    current[rng.choice(2048, size=193, replace=False)] ^= np.uint16(0x7F01)
    bitmap, literals, words = encode_word_bitmap(current, predictor)
    reconstructed = decode_word_bitmap(predictor, bitmap, literals, words)
    assert np.array_equal(reconstructed, current)
    assert bitmap.nbytes == 256
    assert literals.size == 193


def test_word_bitmap_all_equal_and_all_changed():
    predictor = np.arange(32, dtype=np.uint16)
    same = predictor.copy()
    bitmap, literals, words = encode_word_bitmap(same, predictor)
    assert np.array_equal(decode_word_bitmap(predictor, bitmap, literals, words), same)
    assert literals.size == 0
    changed = np.bitwise_xor(predictor, np.uint16(0xFFFF))
    bitmap, literals, words = encode_word_bitmap(changed, predictor)
    assert np.array_equal(decode_word_bitmap(predictor, bitmap, literals, words), changed)
    assert literals.size == 32
