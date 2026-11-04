# Sampled source-slot retention

## Question

Does transport-only source retention explain the slow Ouro fine-Token run?
That candidate uses Prefix3840/Suffix256 at nominal 6.356 requests/s.
The source has two compute slots and zero additional draining slots.
The native transport retains source storage until its required completion and ACK events.

The diagnostic joins admission samples with the latest preceding P-owner snapshot.
The maximum permitted snapshot age is one second.
The observation window extends from measurement start to the last scheduled arrival.
Only waiters with the sole refusal reason `PREFILL_NO_SLOT` qualify as D-credit-clear slot pressure.
The counts below describe samples, not elapsed-time fractions.

## Measured sample counts

| Ouro heavy-load method | Admission samples | D-clear P-slot samples | Fresh P-owner samples | Compute phase + unretired submission | No compute phase + unretired submission | Transfer-only owner, no unretired submission |
|---|---:|---:|---:|---:|---:|---:|
| Native Atomic PD | 92 | 77 | 77 | 0 | 76 | 0 |
| Loop3 | 92 | 15 | 15 | 12 | 1 | 1 |
| Token3072 | 92 | 37 | 37 | 31 | 4 | 2 |
| Token3840 | 92 | 87 | 87 | 35 | 49 | 3 |

Other fresh samples can contain neither a compute-phase owner nor an unretired submission.
They do not necessarily contain a transferring request.
Therefore, the displayed categories need not sum to the fresh-sample denominator.

## Interpretation

Transport-only source retention is not the dominant recorded state in these samples.
Most slot-pressure samples still contain an unretired model submission.
The host request phase can already say `TRANSFERRING` while that submission remains unretired.
Phase-only counting would misclassify this state.

An unretired submission does not prove that a kernel executes at that instant.
A snapshot without an unretired submission does not independently prove GPU idle time.
Sparse snapshots cannot exclude short transport stalls between samples.
The data do not establish CPU, compute, or DMA as the sole throughput limit.
They do not justify attributing the fine-Token slowdown entirely to empty ACK-held slots.

No source-release rule or draining-slot setting changed after this diagnostic.
No early KV release or ACK bypass occurred.
This is a negative diagnostic result, not an optimization speedup.

## Reproduction

```bash
cd /home/esjung/MLLM-EP-github
/home/esjung/.venvs/looped-pd-abc/bin/python -m \
  poc_flashvep.dynamic_loop_split_serving.source_slot_audit
```

The output is `state/source_slot_sample_audit.json`.
It records the exact input-file hashes and sample denominators.
The original owner snapshots, admission records, and request phases remain unchanged.
