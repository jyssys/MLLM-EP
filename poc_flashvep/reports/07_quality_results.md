# Quality results

Status: `IN_PROGRESS`. No final policy quality claim is available.
Model revision: `dad945cac317da394b390f82c7b40691d8a881ed`.
Protocol audit: `01_baseline_reproduction.md`.

The user changed the current scope to a 128/32-question pilot.
The table below retains the original full-evaluation requirements as unmeasured fields.
Pilot baseline results appear in `01_baseline_reproduction.md` and the [pilot table](../results/refinement_aware_ep_expert_allocation/pilot/baseline_accuracy.csv).
Those results must not fill the full-evaluation rows.

| Dataset | Method | Correct/N | Accuracy | Wilson 95% CI | Official reference | Difference from official | Paired difference from vanilla |
|---|---|---|---|---|---:|---|---|
| GSM8K | Fresh HF reference | NOT_MEASURED_FULL | NOT_MEASURED_FULL | NOT_MEASURED_FULL | 94.24% | NOT_MEASURED | NOT_APPLICABLE |
| GSM8K | P0 physical EP4 | NOT_MEASURED_FULL | NOT_MEASURED_FULL | NOT_MEASURED_FULL | 94.24% | NOT_MEASURED | 0.00 pp self |
| GSM8K | P0-Hooked | NOT_MEASURED_FULL | NOT_MEASURED_FULL | NOT_MEASURED_FULL | 94.24% | NOT_MEASURED | NOT_MEASURED |
| GSM8K | P1 uniform | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | 94.24% | NOT_MEASURED | NOT_MEASURED |
| GSM8K | P2 router-only | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | 94.24% | NOT_MEASURED | NOT_MEASURED |
| GSM8K | P3 refinement-only | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | 94.24% | NOT_MEASURED | NOT_MEASURED |
| GSM8K | P4 EP-only | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | 94.24% | NOT_MEASURED | NOT_MEASURED |
| GSM8K | P5 joint | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | 94.24% | NOT_MEASURED | NOT_MEASURED |
| HumanEval | Fresh HF reference | NOT_MEASURED_FULL | NOT_MEASURED_FULL | NOT_MEASURED_FULL | 86.59% | NOT_MEASURED | NOT_APPLICABLE |
| HumanEval | P0 physical EP4 | NOT_MEASURED_FULL | NOT_MEASURED_FULL | NOT_MEASURED_FULL | 86.59% | NOT_MEASURED | 0.00 pp self |
| HumanEval | P0-Hooked | NOT_MEASURED_FULL | NOT_MEASURED_FULL | NOT_MEASURED_FULL | 86.59% | NOT_MEASURED | NOT_MEASURED |
| HumanEval | P1 uniform | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | 86.59% | NOT_MEASURED | NOT_MEASURED |
| HumanEval | P2 router-only | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | 86.59% | NOT_MEASURED | NOT_MEASURED |
| HumanEval | P3 refinement-only | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | 86.59% | NOT_MEASURED | NOT_MEASURED |
| HumanEval | P4 EP-only | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | 86.59% | NOT_MEASURED | NOT_MEASURED |
| HumanEval | P5 joint | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | 86.59% | NOT_MEASURED | NOT_MEASURED |

The fresh raw files preserve complete input and output token IDs.
They also preserve seeds, parsed answers, NFE, EOS status, and generation caps.
Incomplete evaluations must report their actual denominators separately.
They must not occupy the final full-evaluation rows above.

The historical verified scores appear only in `01_baseline_reproduction.md`.
They do not fill these fresh result rows.
HumanEval accuracy requires sandbox execution, not text matching.
Paired quality intervals must resample questions.

The two-request P0/no-op token and NFE checks passed.
Full benchmark no-op agreement remains `NOT_MEASURED`.
The specification's 0.5 pp quality target remains untested.
