# Token origin and startup recovery

## Scope

This note records the first recovery tests of the new campaign.
These tests do not measure clean serving latency.
The model weights, BF16 precision, and fixed recurrent depths remain unchanged.

## Measured original failure

Both original GPU reproductions failed with the expected error:
`wavefront completed prompt chunks out of order`.

The run names are `a_ouro_original_error8` and `a_huginn_original_error8`.
Their configurations disable the imported-suffix origin candidate.
The complete logs remain in the new campaign namespace.

## Measured next failure

`a_ouro_token_partial_origin_v1_state8` enabled only the suffix-origin candidate.
It completed the initial long-prompt warm-up requests.
It then failed with `ValueError: Illegal exact token prefix`.

The failing call used the short Decode-geometry warm-up.
That warm-up contains 512 prompt tokens.
The static Token plan still requested the measured 3072-token Prefix.
The requested Prefix exceeded the warm-up length.

This is a second startup error.
It is not evidence of measured-request output parity.
It also does not prove that the origin candidate fixes all scheduler cases.

## Ranked hypotheses and tests

1. The absolute primary Prefix causes the short warm-up failure.
   An explicit warm-up-relative Prefix must remove this failure.
   The measured Prefix must remain unchanged.
2. A stale warm-up marker can select an unrelated request.
   A request identifier, seed, phase, and length must all match before remapping.
3. General Prefix validation is wrong.
   A legal full-length Prefix must remain legal without remapping.
   An unrelated short request must retain the original validation failure.

The CPU tests execute the existing `static_plan` validation.
The original 512-token case reproduces the exact second error.
The scoped mapping produces 384/512 for Ouro and 496/512 for Huginn.
The tests retain the primary boundaries and reject unrelated remapping.

## Candidate isolation

The second candidate requires `startup_relative_token_boundary=true`.
It applies only to explicitly identified `short_D_geometry` startup requests.
It retains all graph warm-up phases and fixed-depth work checks.
It does not scale measured-request boundaries.

New run names use `startup_v2`.
The first failed runs remain unchanged.
GPU output, KV, hidden-state, and work gates remain mandatory.
The final qualification ledger records the GPU results separately.

## Reproduction

Run these commands from `/home/esjung/MLLM-EP-github`.

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python -m unittest poc_flashvep.dynamic_loop_split_serving.test_warmup_boundary -v
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.dynamic_loop_split_serving.qualification
```

The second command starts GPU experiments.
It waits for the initial qualification controller to finish.
It uses the existing resource guard and the new campaign deadline.
