"""Logical useful BF16 hidden bytes, not wire metadata or HBM traffic."""
import numpy as np


def account(ids, source, ep=4):
    ids=np.asarray(ids,dtype=np.int64)
    if ids.ndim!=2:raise ValueError('expected token-by-k IDs')
    if ep not in (2,4,8) or not 0<=source<ep:raise ValueError('unsupported EP/source')
    if np.any((ids < -1) | (ids >= 128)):raise ValueError('expert ID outside checkpoint bank')
    histogram=np.bincount(ids[ids>=0],minlength=128)
    destination_rows=np.asarray([np.any((ids>=r*(128//ep))&(ids<(r+1)*(128//ep)),axis=1).sum()
                                 for r in range(ep)],dtype=np.int64)
    byte_vector=destination_rows*2048*2
    return {'source':source,'token_rows':len(ids),'valid_paths':int(histogram.sum()),
            'expert_histogram':histogram.tolist(),'active_experts':int((histogram>0).sum()),
            'distinct_destinations_per_token':np.sum([(ids//(128//ep)==r).any(axis=1) for r in range(ep)],axis=0).tolist(),
            'destination_rows':destination_rows.tolist(),'dispatch_bytes_per_destination':byte_vector.tolist(),
            'combine_bytes_per_destination':byte_vector.tolist(),'logical_dispatch_bytes':int(byte_vector.sum()),
            'logical_combine_bytes':int(byte_vector.sum()),
            'logical_remote_bytes':int(byte_vector.sum()-byte_vector[source]),
            'scope':'useful_hidden_BF16_rows_destination_dedup; metadata_padding_wire_headers_excluded'}
