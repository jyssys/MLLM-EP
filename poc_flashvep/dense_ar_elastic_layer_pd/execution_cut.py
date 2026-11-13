"""Distinguish unfinished suffix queries from ordinary cached Decode queries."""


def destination_suffix_cut(cuts,prefilling,computed,prompt_length):
    partial=[cut for cut,phase,start in zip(cuts,prefilling,computed)
        if phase and cut<36 and start<prompt_length]
    if not partial:return None
    assert len(partial)==len(cuts),'A suffix cannot share this runner with active Full36 Decode'
    assert len(set(partial))==1,'A suffix batch must have one real layer frontier'
    return partial[0]
