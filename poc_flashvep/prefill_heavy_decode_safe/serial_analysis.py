"""Apply frozen anchor metadata to new-length serial analysis only."""
import json
from .common import *


def analyze(name):
    configure()
    from ..loop_width_gemm_validation import cost_analysis
    from ..fluidtoken_loop_split import static_screen
    cost_analysis.ROOT=ROOT;cost_analysis.configure=configure
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    frozen=ROOT/'configs/frozen_SLO.json';text=frozen.read_text();key=cfg['model']+'_L'+str(cfg['length'])
    if key in json.loads(text):return cost_analysis.analyze(name)
    assert cfg.get('campaign_frozen_SLO')
    previous=static_screen.json
    class MetadataView:
        def loads(self,value,*args,**kwargs):
            result=json.loads(value,*args,**kwargs)
            if value==text:result=dict(result,**{key:cfg['campaign_frozen_SLO']})
            return result
        def __getattr__(self,item):return getattr(json,item)
    static_screen.json=MetadataView()
    try:return cost_analysis.analyze(name)
    finally:static_screen.json=previous
