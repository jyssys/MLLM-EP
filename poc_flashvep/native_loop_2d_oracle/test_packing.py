import unittest
from .packing import packed_segments


def legacy(info, tables, layout):
    for d, layer, end in layout:
        for start in range(0, end, info['block_size']):
            size = min(info['block_size'], end - start) * info['token_bytes']
            for base in (info['key_ptr'], info['value_ptr']):
                yield base + tables[d][start // info['block_size']] * info['block_bytes'] + layer * info['layer_bytes'], size, info['device']


def mapping(source, target):
    a, b = [], []
    for address, length, device in source:
        a.extend(range(address, address + length))
    for address, length, device in target:
        b.extend(range(address, address + length))
    assert len(a) == len(b)
    result = dict(zip(a, b))
    assert len(result) == len(a)
    return result


class PackingTests(unittest.TestCase):
    def test_exact_every_byte_mapping(self):
        info = dict(block_size=4, token_bytes=2, layer_bytes=8, block_bytes=32,
                    num_layers=4, key_ptr=0, value_ptr=100000, device=0)
        remote = dict(info, key_ptr=300000, value_ptr=500000)
        tables = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        other = [[25, 21, 18], [10, 30, 22], [23, 27, 24]]
        for n in range(1, 13):
            for c in range(4):
                for k in range(n + 1):
                    layout = [(d, layer, n if d < c else k) for d in range(3) for layer in range(4)]
                    self.assertEqual(mapping(legacy(info, tables, layout), legacy(remote, other, layout)),
                        mapping(packed_segments(info, tables, layout), packed_segments(remote, other, layout)))
                    boundary = [(d, layer, n if d < c else k) for d in range(3) for layer in (1, 2)]
                    boundary += [(0, 0, n), (0, 3, max(0, k - 1))]
                    self.assertEqual(mapping(legacy(info, tables, boundary), legacy(remote, other, boundary)),
                        mapping(packed_segments(info, tables, boundary), packed_segments(remote, other, boundary)))

    def test_rejects_aliases(self):
        info = dict(block_size=4, token_bytes=2, layer_bytes=8, block_bytes=32,
                    num_layers=4, key_ptr=0, value_ptr=10000, device=0)
        with self.assertRaises(ValueError):
            list(packed_segments(info, [[1], [1]], [(0, 0, 1)]))


if __name__ == '__main__':
    unittest.main()
