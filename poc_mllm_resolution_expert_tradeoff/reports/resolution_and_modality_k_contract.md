# Resolution and modality-k contract

## Resolution axis

Every cell starts from the same original RGB image after EXIF transpose.  The
installed Qwen processor performs the only resize.  `R0` uses the frozen Part
1 processor contract; `R05` and `R025` set processor `max_pixels` to 0.5 and
0.25 of the R0 processed area, with the official minimum-pixel floor.  Aspect
ratio, patch16, spatial merge2, placeholder count, `image_grid_thw`, DeepStack
and MRoPE behavior remain under the model processor.  Actual grids and visual
token counts, rather than requested ratios, are written per request.  Equal
grids are marked `DUPLICATE_CONFIG`.

## Expert axis

The current router is always evaluated normally.  Its installed Qwen contract
is a 128-expert softmax router selecting and normalizing top-8 weights.  For
visual rows only, the selected list is truncated to k=6 or k=4 and normalized
again over the retained current weights.  Text, special boundary, and all
generated decode rows remain k=8.  Generation sampling `top_k=20` is unrelated
and unchanged.

The adapter captures vLLM's exact multimodal-row mask at the outer embedding
boundary, applies the same TP sequence shard as the hidden states, and changes
only `BaseRouter` output for those rows.  Valid route IDs remain expert IDs;
unused fixed-width slots are `-1` with exactly zero weight.  DeepEP layout and
the sparse Triton expert path accept this sentinel and do not dispatch or run
it.  Consequently these fields are reported separately:

- logical selected visual pairs;
- invalid fixed-width slots;
- actual valid dispatch assignments;
- backend padding/dummy work when present.

`R0/K8` is an identity path that returns the original route tensors.  A
one-request TRUE EP4 smoke for `R0/K4` observed 70 visual rows and exactly 280
valid plus 280 invalid slots at every one of 48 layers, while a 16-token output
prefix matched K8.  This is execution-path evidence, not a quality result.

## Scope

The primary grid uses one uniform visual k per request.  It does not implement
token saliency, video, basis/SVD compression, decoded-token caching, external
OCR, or a second selector network.

