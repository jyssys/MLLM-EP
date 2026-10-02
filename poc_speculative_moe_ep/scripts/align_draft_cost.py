"""Append-only correction of draft-event association, without refitting models.

Native proposer runs after a target frame and produces the *next* frame's
candidate IDs. Earlier tables copied the current-frame event as draft_ms.
The aligned table makes current/next association explicit. All route features,
target/EP labels and model fits remain identical.
"""
import argparse
import json

import pandas as pd

from poc_speculative_moe_ep.contract import ROOT,write,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('versions',nargs='+');a=p.parse_args()
    for version in a.versions:
        root=ROOT/'artifacts'/version;source=root/'verification_metrics.parquet'
        target=root/'verification_metrics_draft_aligned.parquet'
        if target.exists():raise FileExistsError(target)
        table=pd.read_parquet(source);mapping={}
        for run in table.run.unique():
            previous={}
            for line in (ROOT/'raw'/run/'frames.rank0.jsonl').read_text().splitlines():
                f=json.loads(line)
                if f.get('sampled_tokens') is None:continue
                key=(run,f['wave'],f['forward'])
                mapping[key]=previous.get(f['wave'],0.)
                previous[f['wave']]=f.get('draft_ms',0.)
        table['next_draft_ms']=table['draft_ms']
        table['draft_ms']=[mapping[(r.run,r.wave,r.forward)] for r in table.itertuples()]
        table.to_parquet(target,index=False)
        write(root/'draft_alignment.json',dict(original_sha256=sha(source),aligned_sha256=sha(target),
            primary_verification_table=target.name,
            semantics='draft_ms: proposer event on previous native frame, generating current candidate prefix; next_draft_ms: event after current target frame',
            no_ep_target_label_or_feature_changes=True,no_cost_model_refit=True))
        print(version,target.name,len(table))


if __name__=='__main__':main()
