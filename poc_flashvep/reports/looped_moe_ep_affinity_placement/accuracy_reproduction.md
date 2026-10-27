# Official quality reproduction

IN_PROGRESS: this is a preview, not a completed PoC.

## Complete benchmark scores

| Model | EP | Benchmark | Correct/N or status | Accuracy % | Wilson 95% CI % | Paper % | Delta pp | Reference check |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ifm_ouro | 1 | MMLU_5shot_micro_acc | 8979/14042 | 63.944 | [63.146, 64.734] | 64.030 | -0.086 | WITHIN_DECLARED_TOLERANCE |
| ifm_ouro | 1 | ARC_Challenge_0shot_acc_token | 599/1165 | 51.416 | [48.546, 54.277] | 51.330 | 0.086 | WITHIN_DECLARED_TOLERANCE |
| ifm_ouro | 1 | HellaSwag_0shot | UNAVAILABLE_OFFICIAL_DISTRIBUTION | NOT_AVAILABLE | [NOT_AVAILABLE, NOT_AVAILABLE] | 54.820 | NOT_AVAILABLE |  |
| ifm_huginn | 1 | MMLU_5shot_micro_acc | 9379/14042 | 66.792 | [66.009, 67.567] | 66.810 | -0.018 | WITHIN_DECLARED_TOLERANCE |
| ifm_huginn | 1 | ARC_Challenge_0shot_acc_token | 592/1165 | 50.815 | [47.947, 53.679] | 50.470 | 0.345 | WITHIN_DECLARED_TOLERANCE |
| ifm_huginn | 1 | HellaSwag_0shot | UNAVAILABLE_OFFICIAL_DISTRIBUTION | NOT_AVAILABLE | [NOT_AVAILABLE, NOT_AVAILABLE] | 55.100 | NOT_AVAILABLE |  |
| foil1 | 1 | lambada_standard_0shot_acc | 1632/5153 | 31.671 | [30.415, 32.954] | 31.830 | -0.159 | WITHIN_DECLARED_TOLERANCE |
| foil1 | 1 | lambada_openai_0shot_acc | 2056/5153 | 39.899 | [38.570, 41.243] | 39.980 | -0.081 | WITHIN_DECLARED_TOLERANCE |
| foil1 | 1 | xwinograd_en_0shot_acc | 1603/2325 | 68.946 | [67.035, 70.795] | 68.820 | 0.126 | WITHIN_DECLARED_TOLERANCE |
| ifm_ouro | 4 | MMLU_5shot_micro_acc | 8979/14042 | 63.944 | [63.146, 64.734] | 64.030 | -0.086 | WITHIN_DECLARED_TOLERANCE |
| ifm_ouro | 4 | ARC_Challenge_0shot_acc_token | 599/1165 | 51.416 | [48.546, 54.277] | 51.330 | 0.086 | WITHIN_DECLARED_TOLERANCE |
| ifm_ouro | 4 | HellaSwag_0shot | UNAVAILABLE_OFFICIAL_DISTRIBUTION | NOT_AVAILABLE | [NOT_AVAILABLE, NOT_AVAILABLE] | 54.820 | NOT_AVAILABLE |  |
| ifm_ouro | 8 | MMLU_5shot_micro_acc | 8979/14042 | 63.944 | [63.146, 64.734] | 64.030 | -0.086 | WITHIN_DECLARED_TOLERANCE |
| ifm_ouro | 8 | ARC_Challenge_0shot_acc_token | 599/1165 | 51.416 | [48.546, 54.277] | 51.330 | 0.086 | WITHIN_DECLARED_TOLERANCE |
| ifm_ouro | 8 | HellaSwag_0shot | UNAVAILABLE_OFFICIAL_DISTRIBUTION | NOT_AVAILABLE | [NOT_AVAILABLE, NOT_AVAILABLE] | 54.820 | NOT_AVAILABLE |  |
| ifm_huginn | 4 | MMLU_5shot_micro_acc | 9379/14042 | 66.792 | [66.009, 67.567] | 66.810 | -0.018 | WITHIN_DECLARED_TOLERANCE |
| ifm_huginn | 4 | ARC_Challenge_0shot_acc_token | 592/1165 | 50.815 | [47.947, 53.679] | 50.470 | 0.345 | WITHIN_DECLARED_TOLERANCE |
| ifm_huginn | 4 | HellaSwag_0shot | UNAVAILABLE_OFFICIAL_DISTRIBUTION | NOT_AVAILABLE | [NOT_AVAILABLE, NOT_AVAILABLE] | 55.100 | NOT_AVAILABLE |  |
| ifm_huginn | 8 | MMLU_5shot_micro_acc | 9379/14042 | 66.792 | [66.009, 67.567] | 66.810 | -0.018 | WITHIN_DECLARED_TOLERANCE |
| ifm_huginn | 8 | ARC_Challenge_0shot_acc_token | 592/1165 | 50.815 | [47.947, 53.679] | 50.470 | 0.345 | WITHIN_DECLARED_TOLERANCE |
| ifm_huginn | 8 | HellaSwag_0shot | UNAVAILABLE_OFFICIAL_DISTRIBUTION | NOT_AVAILABLE | [NOT_AVAILABLE, NOT_AVAILABLE] | 55.100 | NOT_AVAILABLE |  |
| foil1 | 4 | lambada_standard_0shot_acc | 1632/5153 | 31.671 | [30.415, 32.954] | 31.830 | -0.159 | WITHIN_DECLARED_TOLERANCE |
| foil1 | 4 | lambada_openai_0shot_acc | 2056/5153 | 39.899 | [38.570, 41.243] | 39.980 | -0.081 | WITHIN_DECLARED_TOLERANCE |
| foil1 | 4 | xwinograd_en_0shot_acc | 1603/2325 | 68.946 | [67.035, 70.795] | 68.820 | 0.126 | WITHIN_DECLARED_TOLERANCE |
| foil1 | 8 | lambada_standard_0shot_acc | 1632/5153 | 31.671 | [30.415, 32.954] | 31.830 | -0.159 | WITHIN_DECLARED_TOLERANCE |
| foil1 | 8 | lambada_openai_0shot_acc | 2056/5153 | 39.899 | [38.570, 41.243] | 39.980 | -0.081 | WITHIN_DECLARED_TOLERANCE |
| foil1 | 8 | xwinograd_en_0shot_acc | 1603/2325 | 68.946 | [67.035, 70.795] | 68.820 | 0.126 | WITHIN_DECLARED_TOLERANCE |

IFM MMLU uses the official five-shot, item-weighted metric.
ARC uses all 1,165 eligible four-choice questions and the official `acc_token` scorer.
The 1,172-question generic ARC variant is not this protocol.
Foil uses its own zero-shot paper benchmarks, not IFM reference scores.
The compatibility rule is the larger of 2 percentage points and the Wilson interval half-width.
This rule checks approximate reproduction. It does not establish statistical equivalence.

## Native versus EP task outcomes

| Model | EP | Benchmark | Paired N | Delta pp | Paired bootstrap 95% CI pp | Correct to wrong | Completion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ifm_ouro | 4 | arc_challenge | 1165 | 0.000 | [0.000, 0.000] | 0 | FULL_BENCHMARK |
| ifm_ouro | 4 | mmlu | 14042 | 0.000 | [0.000, 0.000] | 0 | FULL_BENCHMARK |
| ifm_ouro | 8 | arc_challenge | 1165 | 0.000 | [0.000, 0.000] | 0 | FULL_BENCHMARK |
| ifm_ouro | 8 | mmlu | 14042 | 0.000 | [0.000, 0.000] | 0 | FULL_BENCHMARK |
| ifm_huginn | 4 | arc_challenge | 1165 | 0.000 | [0.000, 0.000] | 0 | FULL_BENCHMARK |
| ifm_huginn | 4 | mmlu | 14042 | 0.000 | [0.000, 0.000] | 0 | FULL_BENCHMARK |
| ifm_huginn | 8 | arc_challenge | 1165 | 0.000 | [0.000, 0.000] | 0 | FULL_BENCHMARK |
| ifm_huginn | 8 | mmlu | 14042 | 0.000 | [0.000, 0.000] | 0 | FULL_BENCHMARK |
| foil1 | 4 | lambada_openai | 5151 | 0.000 | [0.000, 0.000] | 0 | PARTIAL_BENCHMARK |
| foil1 | 4 | lambada_standard | 5151 | 0.000 | [0.000, 0.000] | 0 | PARTIAL_BENCHMARK |
| foil1 | 4 | xwinograd_en | 2325 | 0.000 | [0.000, 0.000] | 0 | FULL_BENCHMARK |
| foil1 | 8 | lambada_openai | 5151 | 0.000 | [0.000, 0.000] | 0 | PARTIAL_BENCHMARK |
| foil1 | 8 | lambada_standard | 5151 | 0.000 | [0.000, 0.000] | 0 | PARTIAL_BENCHMARK |
| foil1 | 8 | xwinograd_en | 2325 | 0.000 | [0.000, 0.000] | 0 | FULL_BENCHMARK |

The paired comparison checks prompt hashes before matching outcomes.
Zero changed outcomes do not prove universal equivalence.
The interval measures observed paired outcome changes, not uncertainty over all future workloads.

## Foil held-out language-model loss

| EP | Sequences | Scored tokens | Pure NLL, nat/token | Paper NLL | Native delta | Paired 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1953 | 7997535 | 2.449348 | 2.4492 | 0.000000 | [0.000000, 0.000000] |
| 4 | 1953 | 7997535 | 2.449348 | 2.4492 | 0.000000 | [0.000000, 0.000000] |
| 8 | 1953 | 7997535 | 2.449348 | 2.4492 | 0.000000 | [0.000000, 0.000000] |

An empty loss table means that the GPU loss evaluation did not complete yet.
The official corpus builder checks document identifiers against all 140 training shards.
The rebuilt corpus contains 7,826 documents and 8,000,545 tokens.
The evaluator uses 1,953 complete 4,096-token windows and shifted targets.
Adjacent windows can share source documents. Sequence bootstrap intervals do not remove that dependence.
The paper does not publish a binary hash for its original held-out corpus.
The comparison therefore matches the recipe, not a proven identical token stream.
The 2.499 training loss is not a held-out target.
Native held-out NLL exceeds the declared 0.01-nat reference margin: **False**.
This flag is meaningful only when the native loss row is complete.
An NLL discrepancy requires separate protocol and corpus analysis.
It must not disappear because the downstream anchors match.

## Protocol limitations

The official HellaSwag distribution is unavailable after its repository takedown.
No unofficial mirror substitutes for that distribution.
The full eleven-task IFM paper suite is not complete.
Available tasks use the official entry point and unchanged task specifications.
GSM8K, MATH-500, and DROP were not part of this completed anchor subset.
They receive no invented accuracy result.
Foil retains PyTorch 2.11.0 and Transformers 5.14.1.
Those versions differ from the published requirements file.
See `accuracy_protocol.md`, `accuracy_reference.csv`, and `quality/dataset_manifest.json`.
