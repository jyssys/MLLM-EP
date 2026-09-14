# Window-Diffusion positive control

Official repository revision: `d8bb349e81b74741900049f307184e848006b330`.

Source audit confirms that the implementation ranks still-masked positions in
token order, places the first `active_tokens` in the active set, keeps the
first `window_tokens` plus decoded tokens in the external window, and refreshes
the full state periodically.  The artifact's dense-model speedup is not EP
evidence and will not be attributed to this PoC.

A code-semantic smoke test used `[decoded, MASK, MASK, decoded, MASK, MASK,
MASK, decoded, MASK]`, `window_tokens=4`, and `active_tokens=2`.  It returned
active positions `[1,2]` and the window `[0,1,2,3,4,5,7]`; after position 1 was
accepted, the active masked set advanced to `[1,2,4]`.  This reproduces the
active/buffer/far-field state machine and gives us an independent instrumentation
anchor for “still MASK but not immediately active.”

The supported LLaDA-8B-Base checkpoint was not available locally, so a full
model/quality reproduction is **ENVIRONMENT-LIMITED**.  We did not spend the
primary 100B EP4 budget downloading and tuning a dense positive control.  The
official paper and code already establish that masked tokens can occupy
different computation windows; neither its speedup nor its quality is counted
as measured evidence here.  See the [official repository](https://github.com/vhicrgit/Window-Diffusion)
and [paper](https://arxiv.org/abs/2601.20332).
