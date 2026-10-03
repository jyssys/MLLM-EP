"""Additional decompositions derived only from immutable physical-run tables."""
import argparse
import json
import hashlib
import subprocess
import pandas as pd
from poc_dllm_ep_comm.contract import ROOT,PROJECT,write,sha,DINFER
from poc_dllm_ep_comm.metrics import bootstrap_mean

def main():
    p=argparse.ArgumentParser();p.add_argument('--result',required=True);a=p.parse_args();base=ROOT/'results'/a.result
    target=base/'supplemental_summary.json'
    if target.exists():raise FileExistsError(target)
    df=pd.read_csv(base/'requests.csv');d=df[df.block_length==32];it=pd.read_csv(base/'iterations.csv')
    excess=d.B_actual-d.B_one_shot
    supplements=dict(dead_explained_full_request_excess=bootstrap_mean(d.B_dead_output/excess),
        context_explained_full_request_excess=bootstrap_mean(d.B_context/excess),
        live_explained_full_request_excess=bootstrap_mean((d.B_live_only-d.B_one_shot)/excess),
        pooled_ratios=dict(Raw_Amp=float(d.B_actual.sum()/d.B_one_shot.sum()),
            Residual_Amp=float(d.B_live_only.sum()/d.B_one_shot.sum()),
            Residual_final_position_Amp=float(d.B_live_final_only.sum()/d.B_one_shot.sum())),
        primary_critical_stage_means_ms={k:float(d[k].mean()) for k in ['router_ms','dispatch_ms','expert_ms','combine_ms','denoising_forward_ms','denoising_span_ms','bypass_wall_ms']},
        real_primary_forward_count=int(it[it.request.isin(d.request)].layers.sum()/19),
        zero_accept_iterations=int(d.zero_accept_iterations.sum()),
        secondary_paired_robustness=bootstrap_mean(df[df.block_length==64].Residual_final_positions_Amp),
        annotation='Pooled totals supplementary only; headline is macro request mean with request bootstrap CI')
    write(target,supplements)
    provenance=ROOT/'results/provenance.json'
    if not provenance.exists():
        write(provenance,dict(files={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and
            p.suffix in ('.py','.sh','.diff') and '__pycache__' not in p.parts},
            runtime_patch_sha256=hashlib.sha256(subprocess.check_output(['git','-C',str(DINFER),'diff','--binary'])).hexdigest(),
            external_runtime_unmodified=True,git_base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
            completed_denoising_request_passes={run:len(list((ROOT/'raw'/run).glob('*/*.result.rank0.json'))) for run in ('smoke_v3','main_v1',a.result)},
            tool_note='Full-model request generations and denoising forward calls are different units; neither is multiplied by four physical ranks'))
    print(json.dumps(supplements,indent=2))

if __name__=='__main__':main()
