def test_source_ownership_preserved():
    groups = [[r*8+g*4+i for r in range(4) for i in range(4)] for g in range(2)]
    assert len(set(groups[0]) & set(groups[1])) == 0
    assert sorted(groups[0]+groups[1]) == list(range(32))
    for group in groups:
        for rank in range(4):
            assert all(position//8 == rank for position in group[rank*4:rank*4+4])


def test_natural_whole_buffer_release_cannot_have_row_skew():
    publication = [100.0]*4
    assert max(publication)-min(publication)==0
