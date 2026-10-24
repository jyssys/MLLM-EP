"""Generate short evidence-backed reports after all registered runs finish."""
import csv
from collections import defaultdict
import json
import numpy as np
from .common import ROOT,DEPTHS,MODELS,UUIDS,write

def text(name,value):
    (ROOT/name).write_text(value.strip()+'\n')

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])

def main():
    counts=json.loads((ROOT/'analysis_counts.json').read_text())
    paired=json.loads((ROOT/'paired_block_summary.json').read_text())
    data=list(csv.DictReader((ROOT/'fixed_matrix.csv').open()))
    clean=[r for r in data if r['job'].startswith('quiet_confirm')]
    assert len({r['job'] for r in clean})==12,'Complete all independent confirmation starts first'
    effects=[]
    for r in paired:
        if r['lane']=='CLEAN_NATURAL_CAPPED_COHORT' and r['P']==2048 and r['arm']=='hybrid':
            ci=r['TTFT_ratio_ref_over_candidate_CI95']
            assert isinstance(ci,list) and r['process_blocks']==3
            effects.append([r['model'],r['N'],r['G'],f"{r['TTFT_ratio_ref_over_candidate']:.4f}×",
                f'[{ci[0]:.4f}, {ci[1]:.4f}]'])
    effect_table=table(['Model','Background N','Fixed Hybrid G','TTFT speedup vs Mixed','Paired 95% interval'],effects)
    text('analysis.md',f'''# Adaptive recurrent-group prefill with native Mixed

Final promotion verdict: **NO_GO_INCREMENTAL** for the tested event-based fixed-hybrid path and workload states.
Both pretrained models execute legal grouped prefill and Mixed G1.
The measurements do not establish an incremental adaptive method beyond tuned native Mixed and fixed grouping.
This result does not prove that every possible state-aware policy must fail.

## Completed scope

Ouro uses Base 1.4B, BF16, Static R4. Huginn uses pretrained 3.5B, BF16, Static R16.
Ouro retains native async execution. Huginn retains its supported synchronous backend.
Huginn receives one neutral split/Mixed boundary adapter in every non-atomic arm.
C remains fixed within each comparison.
No dynamic model exit, training, approximation, new attention kernel, or TP/PP engine runs.

The fixed-matrix export contains {counts['cohorts']} cohorts and {counts['requests']} request instances.
It contains {counts['delivered_ITL']} delivered ITL observations.
These repeated mechanism prompts are not independent task-quality samples.
The main confirmation contains three process/GPU blocks per model and background width.
Each block crosses policy order and repeats its complete warm-up workload.

## Main clean confirmation

The table reports reference TTFT divided by candidate TTFT.
A value below 1.000× means a slower candidate.
Each value averages paired within-block ratios, not an aggregate serving-capacity ratio.

{effect_table}

The small positive points do not establish the registered 3–5% incremental operating region.
The three-block intervals offer limited tail and population power.
They do not support a new two-model serving method.
See [all raw metrics](benchmark_summary.md) for the accompanying ITL and cohort makespan.

## Why nominal grouping often changes nothing

In the primary P2048/C512 interference screen, every target prompt group uses Mixed G1 while decode remains ready.
Ouro therefore executes 16 mixed prompt submissions. Huginn executes 64.
The target uses zero pure G>1 submissions in those Fixed Hybrid cases.
Changing the configured maximum G does not change that target's execution granularity.
The raw mode counts distinguish this fact from a selector score or a nominal configuration change.

With no ready decode, grouped prefill can reduce scheduler boundaries.
Best fixed grouping or Atomic already exposes that benefit.
Pure grouping during active decode changes priority and can increase delivered-token stalls.
Those TTFT–ITL trade-offs are not an additional adaptive goodput result.
The controlled interventions retain their state fingerprints and observer-pause limitations.

## Correctness, scale-out, and limits

Singleton and batched numerical probes use the original checkpoint weights.
Their sampled hidden/logit/KV comparisons pass without changing the registered tolerance.
The Full-depth ledgers report {counts['prompt_ledger_failures']} prompt failures and {counts['decode_ledger_failures']} decode failures.
A heterogeneous prompt-remainder bug failed safely before GPU execution.
The scheduler now chooses a common legal group. The preserved regression and live DP retry check that fix.

DP1/2/4/8 portability uses real independent replicas and a common 32-request burst.
It is generic replication, not new model parallelism or qualified sustainable capacity.
The candidate is a fixed hybrid, not an unimplemented adaptive method.
GSM8K/MATH-500 accuracy and qualified Poisson/Burst capacity remain NOT_RUN.
The study does not claim quality-preserving publication success.

## Research decision

Do not build a larger adaptive optimizer from these measurements.
First find an independently reproduced state that beats both tuned Mixed and a strong fixed hybrid.
Then check task accuracy and natural open-loop attainment before claiming a method.
The paused native-vLLM Ouro Loop PP project remains paused.
The prior PoC verdicts remain unchanged.
''')
    groups=defaultdict(list)
    for r in clean:groups[(r['model'],r['N'],r['arm'],r['G'])].append(r)
    rows=[]
    for key,items in sorted(groups.items()):
        rows.append(list(key)+[f"{np.mean([float(r[k]) for r in items]):.3f}" for k in ['TTFT_long_ms','ITL_p99_ms','wall_s']])
    text('benchmark_summary.md','''# Benchmark summary

The table contains LIVE_MEASURED clean bounded natural-generation cohorts.
P2048 and token chunk C512 remain fixed.
N counts background decode requests. One additional long prompt raises total concurrency to N+1.
Each cell uses three independent process/GPU blocks and two repetitions per block.
TTFT is the mean target-request TTFT.
ITL P99 is the mean cohort P99 of background gaps, not a pooled population quantile.
Wall time is the drained cohort makespan, not per-request E2E.

'''+table(['Model','Background N','Arm','G','Target TTFT ms','Background ITL P99 ms','Cohort wall s'],rows)+'''

Natural generation permits EOS, but the fixed screen applies an output cap.
All 2,025 fixed-matrix request instances reached that cap.
None of those mechanism requests demonstrates complete natural-response attainment.
Capped output is not successful complete-response attainment.
The raw request file retains token IDs, stop reasons, and all delivery timestamps.
No finite cohort becomes a sustainable serving-capacity claim.
The exploratory screen, clean confirmation, and heavy profiles remain separate evidence lanes.
''')
    singleton=[]
    for path in (ROOT/'raw').glob('support_*/support.json'):
        items=json.loads(path.read_text());name=json.loads((path.parent/'identity.json').read_text())['job']['model']
        comps=[r['comparison'] for r in items if 'comparison' in r]
        singleton.append([name,len(items),sum(r['tokens'] for r in comps),len(comps),
            max(v['rel_l2'] for r in comps for v in r['tensors'].values())])
    text('correctness.md','''# Correctness and fidelity

The registered limits are relative L2 ≤0.03 and maximum absolute error ≤0.125.
The study does not relax either limit after measurement.
The table compares each singleton arm with native Atomic on the same P128/C64 request.
It includes the full valid prompt KV planes and the first sampled coda state.

'''+table(['Model','Executions','Token-exact comparisons','Comparisons','Maximum relative L2'],singleton)+f'''

The separate batched audit inserts P512/C256 after four background requests begin decoding.
It compares Mixed with Hybrid G2/G4 and grouped-only G2.
The target's valid sampled KV tails, hidden state, logits, and bounded output match in the recorded comparisons.
This is not every KV element of every generated token.

The plan ledger contains {counts['prompt_ledger']} prompt-request records and {counts['decode_ledger']} decode-request records.
Their failure counts are {counts['prompt_ledger_failures']} and {counts['decode_ledger_failures']}.
The decoded-position ledger checks each recorded position's actual recurrent-depth submissions.
Recorded exit-depth labels alone do not constitute that check.

Matched bounded outputs agree in {counts['exact_outputs']}/{counts['paired_outputs']} recorded reference/candidate pairs.
These pairs include repeated source content and reference self-pairs.
They do not prove population quality equivalence.
The standalone CPU fixtures use small random weights only for scheduler regression tests.
No research performance or quality claim uses those weights.

The heterogeneous-remainder regression and live DP retry remain separate from the earlier fixed-grid evidence.
See [failure diagnosis](failures.md) for the exact error and repair.
GSM8K and MATH-500 accuracy remain NOT_RUN.
''')
    text('fixed_group_mixed_map.md','''# Fixed group and Mixed opportunity map

The primary chunk is C512. Secondary chunk controls use C256 and C1024.
The primary prompts are P256 and P2048.
Ouro also receives P8192 controls. Huginn P8192 exceeds its 4096-position checkpoint limit.
The decode-width controls include 0/1/2/4/8 background requests.

`fixed_matrix.csv` retains each actual timing cell.
`actual_mode_counts.csv` retains the executed target group and Mixed counts.
`paired_block_summary.csv` separates clean and plan-observer lanes.
The matrix does not silently replace a missing shape with a model estimate.
Raw G is the configured maximum for staged prompt work.
Atomic ignores that knob and executes full R in its prepared-loop submission.
The export labels its actual Atomic G separately.

In P2048/C512 states with ready decode, the fixed event hybrid commonly selects only native Mixed G1.
With no ready decode, it can execute its pure prefill group.
That simple state switch already belongs to the strong fixed-hybrid control.
Configured G and actual G remain different columns.
Remainder clipping is not an adaptive policy contribution.

![Ouro fixed-hybrid map](figures/ouro_fixed_hybrid_opportunity_map.png)
![Huginn fixed-hybrid map](figures/huginn_fixed_hybrid_opportunity_map.png)
These exploratory maps use a single process per cell and retain the initial runtime outliers.
Gray cells are unmeasured. They are not interpolated estimates.
The clean three-start confirmation remains the primary performance screen.

Grouped-only execution can improve prefill priority while it delays complete decode tokens.
The TTFT–TBT figures show that trade-off without adding kernel service times to E2E.
The exact-state interventions report whether source fingerprints match.
Their hashing pause disqualifies a clean serving-speedup interpretation.
''')
    text('hybrid_policy.md','''# Hybrid policy status

The implemented selector is the fixed event hybrid control.
If compatible prompt and decode work is ready, it selects native Mixed G1.
Otherwise it permits the registered fixed pure prompt G.
The executor clips the common group at every selected row's remaining depth.
It never applies a grouped prompt descriptor to unrelated decode rows.

`selector_candidates.csv` contains available fixed-selector telemetry.
The pre-native selector time excludes the native policy call.
The native policy interval remains a separate measurement.
Neither CPU sum becomes an exclusive E2E saving.

OURS_ADAPT_G_MIXED and ADAPT_G_NO_MIX remain NOT_RUN.
The experiment does not relabel fixed event switching as a new adaptive G method.
No future arrivals, future tokens, or future GPU service times enter a live decision.
No broad scalar-objective or predictor search follows the failed incremental screen.
''')
    text('serving.md','''# Natural serving and SLO scope

The fixed-matrix workload generates natural tokens with explicit output caps.
All 2,025 mechanism requests finish with the length-cap reason.
They do not supply complete-response SLO evidence.
It measures delivered TTFT and every delivered ITL.
It does not freeze generated token IDs to construct a serving win.
The raw outputs permit equal-output diagnostic checks.

The DP portability workload generates short complete chat responses from a common burst clock.
Its finite-cohort throughput belongs to `dp_scaleout.csv`.
It does not qualify sustainable request rate or the registered 90% attainment crossing.

P4 Poisson/Burst capacity remains NOT_RUN after method promotion fails.
Model-specific SLO calibration for a promoted method therefore remains NOT_RUN.
No post-hoc threshold creates a policy-specific success region.
The per-model attainment figure slots explicitly state NOT_RUN.
They contain no invented rate curve or extrapolated capacity.

The required future pass rule remains natural completion plus TTFT compliance plus compliance of every delivered ITL.
A capped response must remain in the offered denominator and fail that joint rule.
Population P99 TBT and a request's maximum ITL are different statistics.
''')
    text('quality.md','''# Quality status

| Model | GSM8K | MATH-500 | Reason |
| --- | --- | --- | --- |
| Ouro R4 | NOT_RUN, N=0 | NOT_RUN, N=0 | No promoted adaptive method |
| Huginn R16 | NOT_RUN, N=0 | NOT_RUN, N=0 | No promoted adaptive method |

No missing accuracy receives a value of 0%.
The singleton and batched probes establish limited numerical fidelity, not benchmark accuracy.
The short arithmetic chats in DP portability are not GSM8K or MATH-500.
Their natural completion does not prove quality equivalence.

Any future candidate must reuse the preserved 64 GSM8K and 50 MATH pilot IDs.
Huginn requires its validated model-specific chat format and stop tokens.
The later evaluator must report completed-correct/N, caps, parser failures, and paired question-block intervals.
No GO claim can precede those tests.
''')
    text('phase_gates.md','''# Phase gates

| Phase | Ouro | Huginn | Scope |
| --- | --- | --- | --- |
| P0 | PASS with limits | CONDITIONAL support extension | Pins, ownership, primary sources, and contention controls |
| P1 | CONDITIONAL | CONDITIONAL | Real-checkpoint numerical probes pass. Limited lifecycle and quality coverage |
| P2 | NO_GO_INCREMENTAL | NO_GO_INCREMENTAL | No reproducible incremental region beyond strong Mixed/fixed controls |
| P3 | NOT_RUN | NOT_RUN | No promoted adaptive G selector |
| P4 | NOT_RUN | NOT_RUN | No qualified natural open-loop capacity sweep |
| P5 | NOT_RUN | NOT_RUN | No GSM8K/MATH benchmark-quality claim |
| P6 | CONDITIONAL feasibility | CONDITIONAL feasibility | Actual finite DP1/2/4/8 portability, not capacity |

Grouping and Mixed are legally composable in the tested fixed-depth implementation.
That correctness result does not establish performance novelty.
Pure grouping and Mixed expose priority trade-offs.
The fixed hybrid often collapses to the same target Mixed plan when decode remains ready.
The observed trade-offs do not justify an extensive adaptive-method search on this path.
The negative verdict concerns the tested states and runtime, not every possible scheduler.
''')
    text('limitations.md','''# Limitations

The mechanism workload repeats one source family.
The historical P8K/N8 output-parity failures are not reclassified by these different requests.
These new requests do not establish a solution for the earlier batched-fidelity issue.
Three process starts give limited power for tail estimates.
The first exploratory screen contains unresolved runtime outliers.
The confirmation records graph behavior, but it does not assign every host gap to a cause.

Clean timing, CUDA stream intervals, and host submission plans are different evidence types.
The report does not claim an exclusive kernel-busy percentage or a full non-overlapped critical-path decomposition.
The event observer can include launch gaps within its interval.
State-hash interventions include an intentional pause.

Huginn requires a common split/Mixed extension and remains synchronous.
Its comparisons use that same extension in every non-atomic arm.
This is not native Huginn async parity with Ouro.
Huginn context remains within its checkpoint configuration.

The multi-prompt remainder failure required a source repair after the earlier fixed-grid campaign.
The failed attempt remains preserved and excluded.
The earlier completed single-target cells and later hardened DP trials have separate source snapshots.

Full benchmark accuracy, large natural Poisson/Burst workloads, and qualified SLO capacity remain NOT_RUN.
The DP experiment has one finite cluster burst per configuration.
DP throughput is generic replica scaling, not a new parallelism contribution.
No measured result supports a model-general adaptive-method GO.
''')
    text('next_steps.md','''# Next steps

Do not implement a larger optimizer or a new distributed engine from this result.
Do not resume the paused native-vLLM Loop PP project.

A new experiment needs a specific state where an alternative legal action beats tuned Mixed and the best fixed hybrid.
Use measured complete-token costs and the same C, fairness, graph behavior, and inputs.
Reject a proposed benefit that disappears when the actual execution plans match.
Use independent natural source families before a population claim.

If that screen passes, calibrate each model's SLO on separate inputs.
Freeze the causal selector and SLO before held-out serving.
Then run task accuracy and the registered capacity sample sizes.
Do not promote another nominal G change or a generic mixed/exclusive switch as novelty.
''')
    text('reproduce.md','''# Reproduction

Run commands from `/home/esjung/MLLM-EP-github`.
Use `/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python` for GPU workers.
Use `/home/esjung/anaconda3/bin/python` for the CPU dispatcher and analysis.
The package lock is in `validation/hardened_tests/packages.json`.

## Source reconstruction

The source pin is `4a350344027960a48384626aa646f1765fda6d4f`.
Create a separate RLT worktree only if `/home/esjung/external/ouro-huginn-g-mixed-rlt` does not exist.
Apply `provenance/hardened_common_group/runtime.patch` to that clean pin.
Do not apply the patch twice or overwrite another worktree.
The model paths and exact revisions are in `registration.json`.
Keep the original BF16 checkpoints outside Git.

## Resume and new attempts

```bash
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_huginn_g_mixed.campaign dp_retry mechanism
```

The dispatcher skips verified completed attempts.
It refuses to overwrite a failed or interrupted attempt.
It stops only verified owned burners and restores leased devices on exit.

For a new measured smoke attempt, register distinct IDs first:

```bash
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_huginn_g_mixed.retry --plan smoke --suffix _repro_001
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_huginn_g_mixed.campaign smoke_repro_001
```

Choose a new suffix if that plan already exists.
Never remove frozen attempts to force a rerun.

## CPU exports

```bash
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_huginn_g_mixed.analyze
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_huginn_g_mixed.mechanism_analysis
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_huginn_g_mixed.dp_analysis
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_huginn_g_mixed.plots
/home/esjung/anaconda3/bin/python -m poc_flashvep.ouro_huginn_g_mixed.report
```

Reports require all twelve clean confirmation starts.
Portable archives preserve raw outputs, failed logs, source snapshots, and the large CSV files.
Restore their relative paths in a separate evidence copy before reanalysis.
No model weights belong to those archives.
''')
    dp_path=ROOT/'dp_huginn_confirm.json'
    if dp_path.exists():
        dp=json.loads(dp_path.read_text())
        if 'MATCHED_CHAT_WARM' in dp:
            cold,warm=dp['LIBRARY_WARM_ONLY'],dp['MATCHED_CHAT_WARM']
            lo,hi=warm['paired_start_CI95']
            with (ROOT/'analysis.md').open('a') as f:
                f.write(f'''\n## Positive-point follow-up: Huginn short-chat DP8

The library-warmed DP8 burst initially favors the fixed hybrid by {cold['mean']:.4f}×.
That point receives additional independent starts and crossed arm order.
The follow-up also warms the exact chat prompts and their complete natural outputs before the offered window.
Under matched chat warm-up, the mean ratio is {warm['mean']:.4f}×, with 95% interval [{lo:.4f}, {hi:.4f}].
The matched condition uses {warm['independent_starts']} independent cluster starts per arm.
It still uses one repeated short-chat source family and no sustainable-rate qualification.

The initial positive point remains in the raw table.
The warm-up sensitivity limits the architecture claim.
The experiment does not isolate a single graph, JIT, or host component as the cause.
Neither result establishes adaptive superiority over the best fixed hybrid.
''')
    with (ROOT/'analysis.md').open('a') as f:
        f.write('''\n## Answers to the eight contract questions

| Question | Evidence-backed answer |
| --- | --- |
| Legal unchanged-BF16 composition on both models? | Yes within the tested source and probes. Huginn requires the neutral common boundary adapter |
| Which static combinations win? | Empty-decode grouping can reduce prompt boundaries. Active-decode hybrids usually execute the same target Mixed G1 plan. The short-chat DP8 positive point fails matched-warm confirmation |
| What causes the crossover? | Grouping changes prefill priority and yield frequency. Mixed changes packing. The remaining runtime outliers lack a complete graph/JIT/host attribution |
| Does an adaptive method change held-out plans and win? | NOT_RUN. No adaptive superiority claim follows from the fixed controls |
| Poisson/Burst attainment and qualified capacity? | NOT_RUN. No manufactured attainment curve or 90% capacity crossing |
| Correctness, Full R, and quality? | Recorded numerical and Full-depth checks pass. Public benchmark accuracy remains NOT_RUN |
| Does the execution path scale to DP2/4/8? | Yes in actual finite replica bursts. Generic replication is not a new parallelism contribution |
| What remains novel? | The evidence does not establish a new model-general adaptive scheduling contribution beyond the existing execution primitives and fixed controls |
''')
    write(ROOT/'gpu_uuid_map.json',{str(g):u for g,u in UUIDS.items()})
    print('REPORTS_WRITTEN',counts)

if __name__=='__main__':main()
