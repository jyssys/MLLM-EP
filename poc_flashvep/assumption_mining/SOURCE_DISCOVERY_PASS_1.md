# Source discovery pass 1: execution ownership and dependency

Read-only audit of the installed vLLM 0.20.0+cu129 / DeepEP 1.2.1 path.

* `vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py:110-162`
  captures `previous_event`, yields from compute to the communication stream,
  passes the event into `get_dispatch_layout` and `dispatch`, and records the
  DeepEP handle per microbatch.
* `deepep_ht.py:183-197` waits on the returned `EventOverlap` before exposing
  expert inputs; `:362-391` repeats the contract for combine and output copy.
* `vllm/v1/worker/ubatching.py` owns compute/communication stream switching and
  GPU event handshakes.  This makes “one worker owns execution ordering” an
  implementation contract, not a theorem about MoE.

**Assumptions generated:** A01--A07, A11, A17--A19, A34--A38.

The existing fixed-shape result is the adversarial control: the same source
contract can create 1.9 s rank-local dispatch waits, but exact route replay
does not reproduce them and direct request tail mass is only 1.09%.  Therefore
the source fact is real, while the counterfactual's economic headroom is not.
