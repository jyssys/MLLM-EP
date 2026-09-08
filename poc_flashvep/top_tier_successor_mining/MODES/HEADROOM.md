# MoDES headroom — keep skipping, quality and E2E separate

## Final GPU resumption supersedes the historical checkpoint below

Final resumption: official1024-question/grid100 frontier completed,436 unique evaluations. Held-out256-question quality and corrected six-condition EP screen completed;three independent GQA-B16 engines do not reproduce the initial large positive point. Known OCR trade-off and port-cost floor do not establish a nontrivial successor. Kimi and new prototype not run.

## Historical audit / release record

The~11.72pp pilot loss is only a quality-only ceiling for that policy/dataset.
It is not efficiency-matched recovery, and the official low-KL calibration may
already address much of it. The full held-out evaluation is deferred.

71.3% or85.3% assignment skipping does NOT imply those percentages of request
latency savings: attention/vision, communication, shape packing and decoding
remain, and the public HF path still computes zero-weight experts.

Clean fast-port E2E baseline: NOT_MEASURED.
Successor quality-matched E2E gain: NOT_MEASURED.
Efficiency-matched quality recovery: NOT_MEASURED.
No strong successor headroom is certified from the partial search.
