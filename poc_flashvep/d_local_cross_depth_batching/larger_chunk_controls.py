"""Check whether short chunks hide a ready-work scheduling advantage."""
import time
from .prepare import *
from .run import execute


def main():
    initialize()
    write(ROOT/'state/larger_chunk_rationale.json',dict(time=time.time(),chunks=[128,248],
        reason='C32 whole-prompt time contains a long solo-prefill tail. Check C128 and a supported 248-row chunk before rejecting all scheduling headroom.',
        numerical_contract='248 prompt rows plus up to 8 decode rows fit the existing 256-row projection tile. The numerical patch stays unchanged.',
        fairness='Each chunk uses the same 512-token budget and 8-entry graph policy in all three arms. Both background streams produce a first token before the local prompt enters.',
        scope='Exploratory micro controls, not an online capacity claim. C32 primary starts remain unchanged.'))
    for model in ('ouro','huginn'):
        cfg=configuration(model)
        cfg.update(name=f'micro_{model}_matched_c248_v1',worker='operator_micro',micro_strict_geometry=True,
            micro_cases=2,micro_grid=[dict(chunk=248,batch=2,aligned=a) for a in (True,False)])
        save(cfg);completed=execute(cfg['name'])
        summary=ROOT/'raw'/cfg['name']/'micro_summary.json'
        valid248=completed and summary.exists() and json.loads(summary.read_text())['all_micro_states_exact']
        for chunk in (128,248):
            if chunk==248 and not valid248:continue
            cfg=configuration(model)
            cfg.update(name=f'opportunity_{model}_c{chunk}_v1',worker='operator_micro',micro_opportunity=True,
                opportunity_chunk=chunk,opportunity_token_budget=512,opportunity_graph_rows=512,opportunity_strict_join=True)
            save(cfg);execute(cfg['name'])


if __name__=='__main__':main()
