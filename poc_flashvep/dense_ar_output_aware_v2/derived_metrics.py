"""Preserve frozen latency formulas and keep controlled replay labels explicit."""
import json
from .common import ROOT,write

def output_contract(cfg):
    if cfg.get('diagnostic_controlled_output_replay'):
        assert cfg.get('ignore_eos') and cfg.get('replay_output_lengths') is not None
        return 'DIAGNOSTIC_FIXED_LENGTH_REPLAY'
    return 'CONTROLLED_FIXED_OUTPUT' if cfg.get('ignore_eos') else 'NATURAL_EOS'

def summarize(name):
    from .bridge import bind
    bind()
    from ..dense_ar_longbench400_pd.metrics import summarize as original
    value=original(name);cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    value['output_contract']=output_contract(cfg)
    write(ROOT/'analysis'/(name+'_metrics.json'),value);return value
