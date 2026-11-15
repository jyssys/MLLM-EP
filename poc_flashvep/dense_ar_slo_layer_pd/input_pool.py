"""Share immutable real prompt lists without changing any request or token."""
import json
from .common import ROOT


def intern(rows):
    canonical={};out=[]
    for row in rows:
        key=row['prompt_id'];ids=row['input_ids']['qwen8']
        if key in canonical:assert canonical[key]==ids,'One prompt ID has inconsistent token arrays'
        else:canonical[key]=ids
        value=dict(row);value['input_ids']=dict(row['input_ids'],qwen8=canonical[key]);out.append(value)
    return out


def load(cfg):
    rows=json.loads((ROOT/'data'/cfg['dataset']).read_text())[:cfg['count']]
    if cfg.get('intern_immutable_prompt_pool'):
        rows=intern(rows)
        # Retire the duplicate loader allocations before any measured arrival.
        import gc
        gc.collect()
    return rows
