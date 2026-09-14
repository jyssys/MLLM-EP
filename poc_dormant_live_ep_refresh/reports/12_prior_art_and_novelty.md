# Prior-art and novelty audit

This audit uses current official papers/code rather than memory.

| System | Removed/reused work | Collision boundary |
|---|---|---|
| [Window-Diffusion](https://github.com/vhicrgit/Window-Diffusion) | Active-window selective computation, buffer KV refresh, far-field token pruning | Establishes dormant-like masked-token locality; it does not isolate routed-expert refresh lifetime or physical EP cost |
| [Elastic-Cache](https://github.com/VILA-Lab/Elastic-Cache) | Attention-aware, layer-aware KV refresh for distant MASK tokens | Strong collision with generic MASK-token caching, but not specifically routed expert dispatch/expert/combine |
| [SureLock](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e82cfb0ee6ce329759d0d3c90fbbccc4-Abstract-Conference.html) | Locks already converged/unmasked positions and skips their query/FFN work | Covers DEAD/STABLE rather than future-dormant still-MASK positions |
| [Epoch](https://arxiv.org/abs/2609.09748) | Compiles diffusion blocks and sends only live/new/refresh-required rows through a fresh EP lane | DEAD work and compiled fresh-row transport are excluded from our claim; the headline must be additional post-Epoch headroom |
| [REFLEX](https://arxiv.org/abs/2608.01784) | Refinement-aware variable expert budget while retaining router ranking | Direct collision if “refresh” degenerates to lower top-k; REFLEX does not reuse normal-top-k routed output across steps |
| [TEAM](https://arxiv.org/abs/2602.08404) | Decoded-state caching plus hot/cold expert activation and speculative decoding | Direct collision for DEAD caching and expert-budget changes; released code is not an EP communication manager |
| [dLLM-Cache](https://github.com/maomaocun/dLLM-cache) | Adaptive transformer/cache reuse for diffusion LMs | Reinforces that generic temporal activation caching is crowded; no routed-EP lifetime/cost isolation |

REFLEX and TEAM change expert/work selection or decoding behavior.  A surviving
candidate here must preserve the normal routed semantics on refresh steps and
change only refresh frequency for a causally safe DORMANT-LIVE subset.  If the
data reduce the idea to generic token caching, top-k reduction, or Epoch's
fresh-lane work, the novelty gate fails regardless of oracle size.

## Adversarial conclusion

The only defensible gap is narrow: still-MASK, future-dormant rows that Epoch
would keep in its fresh lane; fresh attention/shared path/router; normal top-8
semantics on refresh; and fewer physical routed dispatch/expert/combine events.
Window-Diffusion and Elastic-Cache make the semantic premise unsurprising, and
Epoch makes DEAD-row savings unavailable.  REFLEX and TEAM make expert-budget
or routing modifications unavailable as novelty.  Therefore the candidate
must win on **additional post-Epoch routed-EP latency with full-trajectory
quality**, not on dormant-token count.  The measurements below fail that bar.
