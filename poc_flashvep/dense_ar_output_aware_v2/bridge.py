"""Reuse audited algorithms in a new process without editing v1 modules."""
import importlib
import runpy
import sys
from . import common

def bind():
    legacy=importlib.import_module('poc_flashvep.dense_ar_longbench400_pd.common')
    for key in ('ROOT','PREVIOUS','BASE','BRANCH','START','PYTHON','MODEL','MODEL_REVISION','QUIET_LOCK',
                'REVISION','write','digest','preservation','preserve','initialize','resources','time_available','dispatch_allowed'):
        setattr(legacy,key,getattr(common,key))
    from .quality import scores
    quality=importlib.import_module('poc_flashvep.dense_ar_longbench400_pd.quality')
    quality.scores=scores
    if (common.ROOT/'configs/c3_quality_contract_v2.json').exists():
        from .quality_gate import compare
        importlib.import_module('poc_flashvep.dense_ar_longbench400_pd.quality_gate').compare=compare
    return legacy

def main():
    bind();name=sys.argv[1]
    sys.argv=[name]+sys.argv[2:]
    # A guardian holds the same global lease. The explicit module argument also
    # identifies the guarded process to the unchanged legacy dispatcher check.
    if name=='guardian':sys.argv=[name]
    runpy.run_module('poc_flashvep.dense_ar_longbench400_pd.'+name,run_name='__main__')

if __name__=='__main__':main()
