# Decoder-effort controls

## Hidden harness confounds found and fixed

The stock config-42 benchmark silently overwrote both block_length and the CLI
confidence threshold with 32 and 0.9. It also aligned each request's terminal
extent to a fixed bucket of 32. These behaviors made the first variable-B and
threshold-control runs invalid as controls. The isolated dInfer worktree now:

- preserves the explicitly requested B and threshold;
- aligns legacy extents to the B under test;
- accepts an exact common target_total_length;
- supports an opt-in no-EOS-early-stop diagnostic.

Legacy pre-fix rows remain in DECODER_EFFORT_CONTROLS.csv and are labelled
pre-fix threshold overwritten; they are never pooled with corrected runs.

## Corrected matched-effort diagnostic

With threshold 1.0 and EOS early-stop disabled, B still changes NFE. On GSM8K
n=8, aggregate NFE is 224/211/206/206/185 for B=8/16/32/64/128. On HumanEval
it is 281/223/234/263/255. Thus equal endpoint and disabled early stop do not
make decoder effort identical: block semantics change confidence transfer and
the trajectory itself.

This is an important negative control. Clean BCT differences cannot be called a
pure EP effect. The physical effect is therefore isolated separately with
same-device layer replay and matched-M controls.

## Interpretation

- Serving-native results answer what changing B actually does in deployment:
  semantic trajectory, NFE, and systems cost all move together.
- Matched-M results answer the narrower physical question at comparable routed
  rows.
- There is no exact fixed-NFE control without changing the decoder algorithm,
  so no such result is fabricated.
