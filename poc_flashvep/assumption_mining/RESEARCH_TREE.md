# Structural assumption research tree

Root: **Which execution contract is enforced by vLLM/DeepEP despite not being
required by the model or user-visible request?**

| Node family | IDs | Evidence | Headroom | Prior-art / trivial attack | Status |
|---|---|---|---|---|---|
| dependency release | A01--A10 | async receiver, barriers, DP all-reduce | 0--11% known; direct tail 1.09% | async overlap, chunked prefill, selective sync | CLOSED |
| ownership/materialization | A11--A19 | worker-owned streams, host metadata, output copy | metadata <1%; others unknown | vLLM/DeepEP implementation patch | CLOSED/UNKNOWN |
| workload contract | A20--A26 | same HT config and top-k contract | <=2.5% controls; speculative <=11.4% old oracle | Moebius, DA-MoE, TEMPO, SpecMoE | CLOSED |
| scheduling/admission | A27--A32 | DP step and request semantics | <=2.5% request; wave-only large | PROBE/Gimbal/Layered Prefill | CLOSED |
| communication geometry | A33--A38 | A2A matrix, notify/barrier, fixed buffers | <=2.10% config; 1.09% tails | DeepEP/ASAP/ScMoE | CLOSED |
| hardware state | A39--A42 | NVLink mapping, capped SMs, common state | <=0.15% pinning; unknown state | topology/config engineering | CLOSED/UNKNOWN |
| model/request semantics | A43--A50 | top-k, modality, request joins, attention | <=2%; quality gates fail | Capacity-Aware, EPLB, skip/prune | CLOSED/UNKNOWN |

## Selection funnel

* 50 distinct assumptions generated and catalogued.
* 12 source-derived candidates were checked against fresh results.
* 10 reached an analytical score; all measured survivors were below the
  promotion gate or collided with crowded prior art.
* No top-2/3 candidate met the mandatory `>=20%` analytical headroom gate, so
  no GPU perturbation is authorized in this resumed phase.  Running a serving
  sweep would spend GPU time without testing a viable counterfactual.

## Candidate lifecycle rule

`UNTESTED` means the source exposed a choice but no direct mass was measured;
it is not a positive result.  `DROP` means a fresh request-level control or
analytical bound killed the paper-sized opportunity.  A future branch may
promote an UNKNOWN only after adding a low-overhead request join.
