"""Run the exact archived preparation guard on a minimal mixed-row fixture."""
import ast
import copy
import json
import time
import types
import numpy as np
from .common import ROOT,PREVIOUS_SOURCE,initialize,write,digest


def original_preparation():
    path=PREVIOUS_SOURCE/'partial_axis.py';tree=ast.parse(path.read_text())
    candidates=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='prepared_inputs']
    assert len(candidates)==1
    function=copy.deepcopy(candidates[0]);unit=ast.Module(body=[function],type_ignores=[])
    state=types.SimpleNamespace(imports={'partial-request':'source-request'})
    namespace=dict(me=state,inputs=lambda *_:dict(original_prepare_called=True))
    exec(compile(ast.fix_missing_locations(unit),str(path),'exec'),namespace)
    return namespace['prepared_inputs'],path


def reproduce():
    function,path=original_preparation()
    batch=types.SimpleNamespace(num_reqs=2,is_prefilling_np=np.asarray([True,False]),
        req_ids=['partial-request','decode-request'])
    began=time.monotonic()
    try:function(None,batch,None)
    except AssertionError as error:
        assert str(error)=='Do not silently mix partial-layer prefills with full-model decode'
        return dict(reproduced=True,error=str(error),seconds=time.monotonic()-began,
            exact_source=str(path),source_SHA256=digest(path),
            fixture=dict(actual_requests=2,prefilling=[True,False],imported_partial_requests=['partial-request']),
            evidence='CPU_MINIMAL_REPRODUCTION',
            scope='Executes the exact archived prepared_inputs body up to its real assertion. '
            'The input metadata fixture replaces engine setup only. This is not GPU numerical correctness or a fixed adapter performance result.')
    raise AssertionError('The exact mixed-row symptom was not reproduced')


def main():
    initialize();result=reproduce();write(ROOT/'analysis/C3_original_mixed_row_minimal_repro.json',result)
    print('RED_MIXED_ROW_ASSERTION',result['error'],flush=True)
    raise SystemExit(1)


if __name__=='__main__':main()
