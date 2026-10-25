# Measurement and intervention contract

## Scope

The experiment uses exactly three checkpoints.
`model_pins.json` records their immutable revisions.
Each model forward uses one visible H100.
Independent workers share no model state.

The native IFM loader validates artifact files, tensor shapes, and dtypes.
The Foil adapter loads the official remote code from `Foil-1`.
All learned parameters remain BF16 and GPU-resident.
The experiment does not change the published recurrence counts.

## Workloads

The sampling seed is 20261026.
Each domain has 8 smoke examples, 32 calibration examples, and 128 held-out examples.
The language sample has four source groups.
These groups cover factual, narrative or conversational, procedural, and code material.
The source labels do not certify each document's genre.

The language source is `NeelNanda/pile-10k` at revision `127bfedcd5047750df5ccf3a12979a47bfa0bafa`.
GSM8K uses revision `740312add88f781978c0658806c59bc2815b9866`.
MATH-500 uses revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`.
The same raw text reaches all three model tokenizers.
Token IDs and actual lengths can differ between models.

Held-out means separate from this experiment's calibration set.
Pretraining overlap was not audited.
No claim of pretraining non-overlap is available.

## P1 alignment

A comparison key contains the physical MoE site, token position, and recurrent pass.
Prelude and coda sites never enter temporal recurrence denominators.
Four position windows supply at most 32 sampled positions per forward.
The last position is present for generation analysis.
Spatial statistics use only consecutive token positions.

The input observer records the residual before expert normalization.
It also records the actual normalized expert input.
The output observer records raw expert outputs before route weighting.
Weighted contributions use the current native route weights.
Cosine similarity removes scale effects.
Relative L2 and norm ratios retain scale effects.
Absolute L2 accompanies relative-error counterexamples.

The position-shuffled control preserves observed expert activation frequencies.
The pass control uses an earlier pass at the same site.
The summary excludes its degenerate first-pass pairing.
Gap 1, 2, 4, and 8 statistics remain separate.
Generation statistics use the last position of each actual recomputed prefix.

## P2 interventions

O1 ranks baseline weighted-output norms separately at each site and pass.
It then omits selected token-expert edges during another full-model forward.
The default arm does not renormalize surviving route weights.
Named renormalization arms test that separate change.
O1 is a heuristic hindsight policy, not an optimal quality oracle.

O2 selects matched-expert reuse opportunities from baseline output errors.
The changed forward reuses its own prior-pass outputs.
It does not inject baseline outputs into the candidate trajectory.
Current route weights multiply cached raw outputs.

C1 tests current normalized top-k weights below a fixed threshold.
It needs no future activation.
C2 tests prior-pass raw output reuse at the same site, token, and expert.
Its optional guard compares current and prior-pass normalized inputs.
The guard does not track the original input of an older reused output.
Repeated reuse can therefore accumulate stale-output error.
This limitation remains part of the tested mechanism.

Each complete model forward resets the expert cache.
Cross-request, cross-position, and cross-site reuse is forbidden.
Across-token exact prefix caching is not part of this study.
That is a separate generic limitation of these cache-free runtimes.

N1 matches O1's realized row count at each site and pass during teacher forcing.
The initial control repeats its random seed by pass across physical sites.
A separate follow-up uses independent site-specific seeds.
Its teacher-forced comparison repeats O1 at the same prefix cap of 512.
The held-out comparison keeps the original 128 examples.
Its calibration follow-up also uses cap 512, unlike the original cap-128 calibration.
The summary does not duplicate the repeated O1 examples.
Free-running N1 uses the same nominal quota.
Its realized work can differ after token and routing divergence.
The report does not present that rollout control as exact matched work.

## Quality

Next-token NLL uses `logits[:, :-1]` and `input_ids[:, 1:]`.
No label tensor reaches the model loss function.
This avoids Foil's unshifted label contract and auxiliary router losses.
KL uses the baseline distribution as its first argument.
NLL has L-1 targets for an input with L tokens.
Logit agreement has L positions.

Every tested variant runs the complete recurrent model again.
Generation tests use free-running greedy tokens, not teacher forcing.
Every oracle generation step constructs its offline selection from that candidate prefix.
Those extra reference forwards make oracle rollout unsuitable as a speed benchmark.

The downstream pilot contains 8 held-out GSM8K and 8 held-out MATH-500 questions per model.
Its cap is 256 new tokens.
GSM8K uses the official IFM eight-shot text format.
MATH-500 uses the declared zero-shot Problem/Answer format.
EOS and registered next-question stops remain active.
The evaluator uses strict extracted-answer matching.
It does not reproduce the official MATH evaluator or a private judge.
Parsing failures, input clipping, and output truncation remain explicit.
This pilot cannot establish a 1 percentage point quality margin.

## Timing

Clean forward timing uses 5 warmups and 20 repeated measurements.
Three independent model processes measure each primary cell.
The primary timing GPU is physical GPU0.
Unused GPUs retain owned burners as requested.
This is a quiet research lane, not a machine-wide idle baseline.

CUDA-event module spans come from separate observer runs.
They include internal launch gaps.
They are not sums of GPU kernel service times.
Observer ratios beyond 10% receive a PARTIAL label.
Attention, router, norm, and output spans remain separate from expert spans.
Unresolved dispatch, combine, and host intervals do not receive invented percentages.

The generation baseline uses fixed 16- and 64-token hardware shapes.
That EOS-disabled measurement is not an accuracy result.
Natural quality generation uses separate runs with real stop conditions.
Generation TPOT always denotes full-prefix recomputation here.

The native-weight microbenchmark uses observed routing geometry at one recurrent site per model.
The corrected Foil measurement prepares GPU offsets before the timer.
The IFM measurement retains its native count handling.
Selection, compaction, routing, cache, and combine remain excluded.
Component speedups are not E2E speedups.

## Statistics and bounds

Quality intervals resample independent sequences 4,000 times.
NLL averages weight sequences by their scored token counts.
Routing intervals preserve each sequence's correlated sites and passes.
Timing intervals resample three process-start means.
Within-frame quantile averages are not pooled token quantiles.

The Amdahl model uses `1 / (1 - f_expert × q)`.
The optimistic version divides event spans by clean wall time.
It assumes proportional time removal and zero selection or cache cost.
It is not a measured speedup or a universal mathematical bound.
The microbenchmark checks the proportional-removal assumption separately.
