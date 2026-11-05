"""Reuse native page/layer packing for arbitrary valid depth-plane prefixes."""


def packed_segments(info, tables, layout):
    ends = {}
    for depth, layer, end in layout:
        if (depth, layer) in ends:
            raise ValueError('Duplicate valid KV plane')
        ends[depth, layer] = end
    block = info['block_size']
    dense = info['layer_bytes'] == block * info['token_bytes']
    used = [value for table in tables for value in table]
    if len(used) != len(set(used)):
        raise ValueError('Native packed descriptors require non-aliased depth blocks')
    for depth, table in enumerate(tables):
        length = max((end for (d, layer), end in ends.items() if d == depth), default=0)
        for start in range(0, length, block):
            runs = []
            for layer in range(info['num_layers']):
                count = min(block, max(0, ends.get((depth, layer), 0) - start))
                if not count:
                    continue
                if dense and count == block and runs and runs[-1][2] == block and runs[-1][0] + runs[-1][1] == layer:
                    first, layers, previous_count = runs[-1]
                    runs[-1] = first, layers + 1, previous_count
                else:
                    runs.append((layer, 1, count))
            for layer, layers, count in runs:
                size = count * info['token_bytes'] if layers == 1 else layers * info['layer_bytes']
                for base in (info['key_ptr'], info['value_ptr']):
                    yield base + table[start // block] * info['block_bytes'] + layer * info['layer_bytes'], size, info['device']
