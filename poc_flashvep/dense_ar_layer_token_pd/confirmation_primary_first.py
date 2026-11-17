"""Run both output lengths' independent repeats before exploratory endpoints."""
import fcntl
import json
import random
from .common import ROOT,SOURCE,QUIET_LOCK,resources,digest,save,write
from .confirmation import wait_file,choose,qualify_suffix,config
from .analysis import metrics
from .run import execute
from .heldout_baselines import SEEDS


def perform(contract,qps,seed,values):
    methods=['Token','Static','Dynamic'];random.Random(seed).shuffle(methods)
    for method in methods:
        for name,expected in contract['worker_hashes'].items():
            assert digest(SOURCE/name)==expected,'Frozen worker changed: '+name
        cfg=config(contract,method,qps,seed)
        if not execute(save(cfg)):
            continue
        value=metrics(cfg['name']);assert value['output_gate']['passed']
        values.append(value);write(ROOT/'analysis/heldout_candidate_results_primary_first.json',values)


def main():
    values=[];contracts={}
    try:
        contracts[16]=json.loads((ROOT/'state/candidate_confirmation_freeze_O16.json').read_text())['contract']
        qualify_suffix(contracts[16]['selected']['Token']['representative']['token_suffix'])
        for seed in SEEDS:
            perform(contracts[16],6.4,seed,values)
        wait_file(ROOT/'analysis/output64_axis_frontier.json',12)
        wait_file(ROOT/'analysis/flow_homogeneous64_discovery.json',2)
        wait_file(ROOT/'analysis/adaptive_output64_flow_discovery.json',2)
        wait_file(ROOT/'analysis/output64_fraction_token_refinement.json',5)
        contracts[64]=choose(64)
        qualify_suffix(contracts[64]['selected']['Token']['representative']['token_suffix'])
        for qps in (4.8,6.4):
            for seed in SEEDS:
                perform(contracts[64],qps,seed,values)
        for cap,rates in ((16,(2.,10.,4.8)),(64,(2.,10.))):
            for qps in rates:
                perform(contracts[cap],qps,SEEDS[0],values)
        write(ROOT/'state/candidate_confirmation_complete.json',dict(
            primary_first=True,completed_entries=len(values),
            scope='Same42 immutable candidate configurations and same seeds. Only admission order changes. Completed receipts are reused without overwriting evidence.'))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__=='__main__':
    main()
