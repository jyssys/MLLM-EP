"""Keep the real P token and import only valid original-prompt blocks."""


def valid_prefix_blocks(local,remote,prefix_tokens,block_size):
    assert prefix_tokens>0 and block_size>0 and prefix_tokens%block_size==0
    expected=prefix_tokens//block_size
    assert len(remote)==expected,'Source cannot advertise an uncomputed new-token block'
    assert expected<=len(local)<=expected+1,'Destination must retain the new-token reservation'
    return list(local[:expected])


def continuation_input(row,token,cap,prefix_tokens,kv):
    assert cap>=2 and len(row['input_ids']['qwen8'])==prefix_tokens
    extended=dict(row,input_ids=dict(row['input_ids'],qwen8=list(row['input_ids']['qwen8'])+[token]))
    return extended,cap-1,dict(kv,dense_source_prefix_tokens=prefix_tokens)
