"""Summarize measured fidelity without promoting the failed state-space gate."""
import csv
import numpy as np
from poc_ouro_reasoning_fidelity_multigpu.contract import *

def main():
    rows=read(ROOT/'results/fidelity_comparisons.json')
    summary=read(ROOT/'results/fidelity_summary.json')
    diagnostic=read(ROOT/'results/calibration_diagnostic_summary_v2.json')
    flat=[dict(id=r['id'],split=r['split'],TP=r['TP'],rank=r['rank'],step=r['step'],
        logits_relative_L2=r['logits']['relative_L2'],hidden_relative_L2=r['hidden']['relative_L2'],
        KV_relative_L2=r['KV']['relative_L2'],KV_worst_slot_relative_L2=r['KV']['worst_slot_relative_L2'],
        JSD_nats=r['logits']['JSD_nats'],top1_equal=r['logits']['top1_equal']) for r in rows]
    with (ROOT/'results/fidelity_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]))
        writer.writeheader();writer.writerows(flat)
    table=[]
    for tp in [1,2,4]:
        chosen=[r for r in rows if r['TP']==tp]
        confirmation=[r for r in chosen if r['split']=='confirmation']
        table.append(f"| TP{tp} | {100*max(r['hidden']['relative_L2'] for r in chosen):.3f}% | {100*max(r['KV']['relative_L2'] for r in chosen):.3f}% | {max(r['logits']['JSD_nats'] for r in chosen):.8f} | {max(r['logits']['JSD_nats'] for r in confirmation):.8f} |")
    norm=[]
    for r in diagnostic['norm_ablation']:
        norm.append(f"| {r['mode']} | {r['step']} | {100*r['hidden']['relative_L2']:.3f}% | {100*r['KV']['relative_L2']:.3f}% | {r['logits']['JSD_nats']:.10g} |")
    text='''# HF and native runtime fidelity

Verdict: `REFERENCE_RUNTIME_PARTIAL`.
The original strict state-space planning gate does not pass.
No tolerance changes after confirmation analysis.
The functional next-token check and selected same-configuration repeats pass within their stated scope.
These results do not establish exact HF equivalence.

## Scope

The study uses eight frozen AIME prompts and two metadata-ordered MATH prompts.
Four AIME prompts calibrate limits.
The other four AIME prompts and two MATH prompts confirm them.
These are held out from tolerance calibration, not fresh unseen benchmark questions.
The teacher checkpoints are 0, 1, 4, 32, 128, and 1,024 cached steps.
Every engine receives the same frozen prompt IDs and teacher prefix.

The reference uses the pinned remote code with Transformers 4.57.1 and SDPA.
The author-recommended 4.54.1 combination fails the pinned cache constructor.
That failure remains in the evidence.
The native engine uses BF16, FlashAttention3, ordinary CUDA Graphs, and TP1/2/4.
Physical GPU masks are 4, 4–5, and 4–7 respectively.
The GPU6 operator diagnostic is not the GPU4 primary TP1 baseline.

## Numerical results

Each row below gives maxima, not confidence intervals.
Rank-local logits replicate within a TP group.
Ranks, layers, and token checkpoints are not independent serving samples.
KV metrics compare selected written positions and coordinates, not every cache byte.

| Native layout | Max hidden relative L2 | Max sampled-KV relative L2 | Max JSD, all states, nats | Max JSD, confirmation, nats |
|---|---:|---:|---:|---:|
'''+ '\n'.join(table)+'''

The fixed functional JSD limit is 0.005 nats.
All confirmation states satisfy that distribution-level limit.
The initial hidden, logit, and KV planning ceilings do not pass calibration.
The calibrated thresholds and their failed ceiling flags remain unchanged.
The report does not replace this failure with a numerical PASS.

All 126 selected numerical fields repeat bitwise within the same native configuration.
They comprise 42 state/rank pairs and three fields per pair.
Only one calibration problem supplies that numerical-repeat test.
Short natural-output repeats also match in the compared conditions.
Cross-backend greedy outputs can diverge.
The raw first-divergence indices remain available.

## Calibration diagnosis

All 389 loaded TP1 parameters match the pinned BF16 checkpoint tensors bitwise.
The comparison includes stacked QKV and gate/up weights.
The diagnostic changes no checkpoint tensor.

The following ablation keeps eager execution and GPU6 fixed.
The HF-operator arm changes the norm path only.
It is not a benchmark baseline or a performance candidate.

| Norm path | Cached step | Hidden relative L2 | Sampled-KV relative L2 | JSD, nats |
|---|---:|---:|---:|---:|
'''+ '\n'.join(norm)+'''

The late mismatch decreases when the norm path changes.
The earlier checkpoints do not improve uniformly.
This result provides partial operator-path attribution, not a complete error explanation.
The primary compiled runtime remains unchanged.

Three calibration first-divergence probes reproduce both engines' prior greedy choices.
Their Top-2 margins are 0 through 0.125 logits.
Their JSD values are approximately 0.00049 through 0.00050 nats.
All three preserve the same Top-5 token set.
This evidence is consistent with small numerical changes near greedy ties.
It does not prove equal complete-output quality across backends.
The diagnostic native caps differ from the earlier natural cap.
The executed source snapshot preserves that limitation.

The initial analysis used reverse argsort at exact ties.
A regression exposed its disagreement with greedy argmax.
The corrected analysis uses stable descending order.
The old derived summary remains preserved and is not authoritative.
No raw output, benchmark score, or tolerance changes because of that correction.

## Natural-output evidence and limits

Two longer MATH probes end with EOS under HF and native TP1/2/4.
All eight completed answers match their references under the unchanged MATH grader.
Those are two independent problems, not a MATH-100 accuracy result.
The AIME 256-token and 2,048-token probes remain capped.
They provide no completed AIME benchmark accuracy.
The separate official-card arithmetic probe ends with EOS and states x=4.

The pre-final-norm native state is NOT_CAPTURED under the compiled primary path.
The report compares post-norm hidden state, full pre-sampler logits, and sampled KV.
The diagnostic hooks copy tensors to the CPU.
Their wall times cannot serve as headline latency.
Preferred-version compatibility, strict state closeness, functional agreement, and quality remain separate gates.

A small public-quality diagnostic uses the unchanged native model.
P1 remains the source-selected primary regardless of the diagnostic score.
No new architecture follows from this partial-reference result.

Evidence: `results/fidelity_summary.json`, `results/fidelity_metrics.csv`, and `results/calibration_diagnostic_summary_v2.json`.
See `results/fidelity_natural_grading.json` for the bounded natural-answer checks.
'''
    replica=read(ROOT/'results/TP_replica_state_audit.json')
    drift=read(ROOT/'results/TP_hidden_rank_drift_summary.json')
    fields={name:[r for r in replica['records'] if r['field']==name] for name in ['logits','hidden']}
    exact={name:sum(r['all_ranks_bitwise_equal'] for r in values) for name,values in fields.items()}
    text+=f'''

## Cross-rank replication check

The stored TP2/TP4 trace contains {len(fields['logits'])} problem/checkpoint groups per field.
Full-logit groups match bitwise across ranks in {exact['logits']}/{len(fields['logits'])} comparisons.
Hidden states match bitwise in {exact['hidden']}/{len(fields['hidden'])} groups.
The other {drift['non_bitwise_groups']} groups have maximum relative L2 of {drift['maximum_relative_L2']:.10f}, or approximately {100*drift['maximum_relative_L2']:.4f}%.
The maximum absolute hidden difference is {drift['maximum_absolute_difference']}.

These two maxima can come from different groups.
The responsible operator is NOT_CAPTURED.
The report does not attribute this difference to a specific collective.

These comparisons reuse {replica['independent_teacher_problems']} teacher problems and six checkpoints at TP2 and TP4.
They do not add independent serving samples.
Cross-rank hidden differences and the earlier same-rank repeat check measure different properties.
Equal full logits do not establish bitwise hidden-state replication.
The overall reference verdict remains PARTIAL.

Evidence: `results/TP_replica_state_audit.json` and `results/TP_hidden_rank_drift_summary.json`.
'''
    (ROOT/'hf_vllm_fidelity.md').write_text(text)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=ROOT/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    steps=[0,1,4,32,128,1024]
    for tp in [1,2,4]:
        for axis,key,group in [(axes[0],'relative_L2','hidden'),(axes[1],'JSD_nats','logits')]:
            values=[max(r[group][key] for r in rows if r['TP']==tp and r['step']==step) for step in steps]
            axis.plot(range(6),[max(v,1e-12) for v in values],marker='o',label=f'TP{tp}')
            axis.set_xticks(range(6),steps);axis.set_xlabel('Cached step');axis.grid(alpha=.25)
    axes[0].axhline(.1,color='black',linestyle='--',label='Original hidden planning ceiling')
    axes[0].set_ylabel('Maximum hidden relative L2')
    axes[1].axhline(.005,color='black',linestyle='--',label='Frozen functional JSD limit')
    axes[1].set_yscale('log');axes[1].set_ylabel('Maximum next-token JSD, nats')
    for axis in axes:axis.legend(fontsize=8)
    fig.suptitle('REFERENCE_RUNTIME_PARTIAL: strict state ceiling remains failed')
    fig.tight_layout();fig.savefig(figures/'fidelity_state_vs_distribution.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    main()
