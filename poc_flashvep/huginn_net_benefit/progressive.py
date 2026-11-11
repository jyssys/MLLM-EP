"""Exact plane accounting for optional early KV publication."""


def delta_layout(layout, published):
    """Return unsent valid planes. Reject a frontier behind published state."""
    current = {(depth, layer): end for depth, layer, end in layout}
    if len(current) != len(layout):
        raise ValueError('Duplicate KV plane')
    for key, end in published.items():
        if current.get(key, -1) < end:
            raise ValueError('The final frontier excludes an already published KV plane')
        if current[key] != end:
            raise ValueError('A published plane changed its valid token range')
    return [(depth, layer, end) for depth, layer, end in layout if (depth, layer) not in published]


def disjoint_publication(layout, published):
    for depth, layer, end in layout:
        key = depth, layer
        if key in published:
            raise ValueError('A KV plane must not be published twice')
        published[key] = end
