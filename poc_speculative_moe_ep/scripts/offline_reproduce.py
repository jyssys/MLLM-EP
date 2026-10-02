"""Reproduce grouped-CV numbers from the published ledger without raw/GPU."""
import argparse
import json
import numpy as np
import pandas as pd
from poc_speculative_moe_ep.contract import ROOT
from poc_speculative_moe_ep.scripts.analyze import cost_models


def main():
    p=argparse.ArgumentParser();p.add_argument('--version',default='phase_b_v2');a=p.parse_args()
    root=ROOT/'artifacts'/a.version
    table=pd.read_parquet(root/'verification_metrics_draft_aligned.parquet')
    expected=json.loads((root/'cost_model_results.json').read_text());actual=cost_models(table)
    for family in ('M0','M1','M2'):
        for target in ('ep_comm_ms','moe_ms','target_forward_ms'):
            for metric in ('mae','r2'):
                assert np.isclose(actual[family][target][metric],expected[family][target][metric],atol=1e-9), (family,target,metric)
    assert actual['folds']==expected['folds']
    print(json.dumps(dict(status='PASS',requests=table.request_id.nunique(),rows=len(table),
        ep_mae_reduction_pct=actual['ep_mae_reduction_pct'],ep_r2_increment=actual['ep_r2_increment'])))


if __name__=='__main__':main()
