# Paper assumption matrix

This is an adversarial prior-art audit, not a literature survey of methods.
The question for each line is: **which execution assumption did the paper
break, and which equally fundamental assumption did it leave intact?**

| Work / family | Assumption it breaks | Assumption not questioned | Consequence for a new candidate |
|---|---|---|---|
| DeepEP / DeepEP HT | expert communication can be asynchronous and overlap compute | a dispatch/combine contract still exposes a per-invocation dependency and fixed rank topology | a candidate must be beyond generic A2A overlap |
| vLLM V1 EP integration | a serving step can be coordinated by a common scheduler and DP descriptor | one model execution contract per step; worker owns execution | scheduler/worker ownership must show new critical mass |
| DA-MoE / fused-MoE load-aware systems | per-expert workload shape matters to kernel cost | all routed assignments remain semantically required | histogram-independent structural work is the only open angle |
| TEMPO / load-balancing representations | replica/load state can be represented and balanced over time | current per-token execution follows the selected placement | placement-free counterfactual must be distinguished |
| Moebius / HAP | parallelism/topology can be selected by workload | a request/window has one effective execution mode | simple TP↔EP switching is closed here |
| PROBE / Gimbal / ExpertPlex | request/batch composition can be used for serving decisions | request critical work is still scheduled as a conventional batch | a new candidate needs more than locality/admission |
| ELDR / Semantic Parallelism | expert-locality can influence batching | all selected experts/contributions are executed | locality-only rebatching is closed |
| ScMoE / FarSkip-Collective | communication/computation overlap or skipping is exploitable | exact downstream dependency is respected | generic overlap and skipping are not novelty |
| Layered/Chunked Prefill / ZeRO-Prefill | prefill can be staged or layered | token/state dependencies and collective contract remain | avoid rebranding chunked scheduling |
| SpecMoE | approximate expert work can draft tokens and later verify | verification targets token generation, not a generic downstream EP contract | affinity substitution is a baseline, not novelty |
| Capacity-Aware-MoE / MACS / SERE | capacity/routing can be changed to reduce overload | token routing remains the semantic unit | no rerouting/drop policy in this branch |
| EPLB | expert placement/replication can follow persistent load | each invocation still dispatches to a fixed placement | placement assumptions need new evidence, not a placement tweak |
| ASAP / asynchronous MoE systems | synchronization/overlap policy can be adjusted | collective correctness and model-level contribution are preserved | debt/state candidate must be orthogonal to known overlap |
| Sparse/conditional-execution literature | not every expert need be evaluated | selected expert set is chosen before execution | partial execution is crowded and quality-bound |

## Adversarial conclusions

* The strongest unclosed source fact is ownership/materialization: vLLM's
  worker builds metadata and DeepEP owns asynchronous receiver dependencies.
  Existing systems generally optimize that contract, rather than asking whether
  the ownership boundary itself is necessary.  However the measured metadata
  mass is too small and the communication-tail direct mass is 1.09%.
* The tempting “cross-rank fanout geometry” idea is not a new axis after
  controlling rank load in real online data; it is therefore not promoted.
* Any claim based only on wave makespan, per-rank duration, or a synthetic
  operator replay is marked `CORRELATIONAL_ONLY` until request joining exists.
* A new structural direction would need a repeated >=15--20% request-level
  counterfactual and survive simple sync/backend/config attacks. No such
  candidate is currently evidenced.
