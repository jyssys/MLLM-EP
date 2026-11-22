"""Load the current controller only after its explicit predecessor checkpoints."""
import importlib
import sys
import time
from .common import ROOT,SOURCE,write,digest,dispatch_allowed

def main():
    module=sys.argv[1];markers=sys.argv[2:]
    assert module in ('c3_campaign','c3_policy','c3_main') and markers
    while not all((ROOT/'state'/m).exists() for m in markers):
        if not dispatch_allowed():
            write(ROOT/'state'/f'{module}_deferred_controller_v2.json',dict(time=time.time(),status='NOT_RUN',reason='DISPATCH_BUDGET_ENDED',markers=markers));return
        time.sleep(5)
    write(ROOT/'state'/f'{module}_deferred_controller_v2.json',dict(time=time.time(),status='STARTING',markers=markers,
        controller_source_SHA256=digest(SOURCE/(module+'.py'))))
    target=importlib.import_module('poc_flashvep.dense_ar_output_aware_v2.'+module);target.run()

if __name__=='__main__':main()
