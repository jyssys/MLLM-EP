"""Verify immutable inheritance before any new model execution."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from .common import *


def main():
    setup()
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()!=BASE:
        raise RuntimeError('The initial checkout must be the declared RCC release')
    from ..rcc_pd_poc import preflight as prior_check
    prior_check.ROOT=ROOT
    prior_check.main()
    record=json.loads((ROOT/'configs/prerequisite_gate.json').read_text())
    assert all(r['status']=='CORRECTNESS_READY' for r in record['models'].values())
    for name in ('poc_flashvep/rcc_pd_poc/worker.py','poc_flashvep/looped_pd_abc/split.py'):
        assert subprocess.check_output(['git','show',BASE+':'+name],cwd=REPO)==(REPO/name).read_bytes()
    record.update(base=BASE,study='Dynamic A, no Native Mixed',
                  new_spec_sha256=digest(REPO/'poc_flashvep/reports/adaptive_recurrence_cut_vs_token_layer_split_poc_spec.md'),
                  initial_git_index_sha256=hashlib.sha256(subprocess.check_output(['git','ls-files','--stage'],cwd=REPO)).hexdigest())
    write(ROOT/'configs/source_contract.json',record)
    print('BOTH_MODELS_INTERNAL_CORRECTNESS_READY; NEW_MIGRATION_GATES_REQUIRED')


if __name__=='__main__':main()
