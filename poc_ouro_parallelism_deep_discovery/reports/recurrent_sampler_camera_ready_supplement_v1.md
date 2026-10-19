# Recurrent sampler: camera-ready source supplement

The [official ICML paper](https://raw.githubusercontent.com/mlresearch/v306/main/assets/geiping26a/geiping26a.pdf) is now readable through an isolated PDF parser.
This supersedes the earlier reader-access limit, not its archived record.
The [source receipt](../artifacts/recurrent_sampler_camera_ready_supplement_v1.json) records the hash and inspected extent.
The PDF itself is not redistributed.

Sections 3.1–3.2 require input injection and decodable intermediate states.
Their latest-state KV sharing evidence concerns Huginn, not unchanged Ouro.
Remark 3.1 makes convergence conditional on contraction and calls that condition conceptual.
It does not establish fixed-R4 or BF16 parity.

Table 1 uses B1 Hugging Face execution with dynamic KV, not optimized distributed Ouro.
Section 5 reports sandboxed A100 measurements.
Appendix A.3 specifies temperature0.2 and top-p0.95.
Its task-quality protocol is not this campaign's greedy protocol.

Equation1 requires every preceding position to pass the exit test.
Algorithm2 instead describes the last passing position without that explicit conjunction.
This is a source-specification ambiguity, not a measured code defect.
Appendix A.2 leaves the proposed batched/device-movable state engine unimplemented.

The final body does not close the campaign's original-gate, depth-KV, physical-cost, or quality obligations.
No candidate changes status.
No model, GPU, or protected outcome enters this source-only supplement.
