# Native MoE versus MLLM transfer

Dense Qwen2.5-32B PP4 and native Qwen3-30B-A3B PP4 have fresh live request
measurements. The latter is a true routed MoE model, but **not expert parallel**:
each owned layer retains all experts, and only tensor dimension sharding is
implemented by this pinned Qwen FusedMoE class. A generic enable-EP flag does not
establish PP×EP. See CODE_AUDIT.md for the actual imports and ownership path.

The sparse native ALP diagnostic records the real predictor, not a substituted
cross-layer heuristic: 3,950 complete four-PP-rank identities, 127 complete
48-layer profiles. Pre-update median relative host prediction error is 8.62%
for extend and 3.64% for mixed; decode's 18.59% has only 13 learning samples.
These numbers do not establish an MLLM failure or an E2E predictor oracle.

Qwen3-VL DeepEP4 common traces are a separate, explicitly labelled transfer
diagnostic, including 120 fresh exact-token-count visual/text request pairs.
No native VL or joint PP×EP request-level result is claimed. A faithful port
would need multimodal position/deepstack handling and real expert ownership,
not merely a CLI flag. Port effort is not a method failure.
