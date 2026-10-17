# Prior boundary

## AsyncDiff

[AsyncDiff](https://arxiv.org/abs/2406.06911) partitions a diffusion model and reuses features from adjacent denoising steps.
Its implementation separates asynchronous execution from stride denoising.
PARPP tests the dependency-relaxation idea on recurrent language-model passes.
It does not assume that language-model passes share diffusion feature dynamics.
No AsyncDiff quality result or speedup transfers to this experiment.

The source audit pins the official repository at `47cae2b021ec14a75eac8c37b14f69baf7f5cd07`.
The saved README and its hash appear in `sources/` and the experiment manifest.

## Ouro

[Ouro-1.4B-Thinking](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking) supplies the pretrained weights and official forward implementation.
The experiment preserves BF16 weights, four recurrent passes, attention masks, RoPE, and independent depth-indexed KV.
The model card describes recurrent output selection. The experiment does not equate output selection with skipped GPU computation.
The card recommends older Transformers but also documents a later cache compatibility fix.
This experiment records the actual pinned source and checks its behavior on Transformers 4.57.1.

## Continuous Depth Batching

[CDB](https://arxiv.org/abs/2608.09444) schedules variable-depth loop execution across requests.
Its queues and lookahead exit decisions address batching and scheduling overhead.
PARPP instead predicts unavailable inputs to relax dependencies within one recurrent pass of one token.
The experiment uses fixed R4. It does not implement a depth policy, reinforcement learning, or a CDB scheduler.
Independent depth KV remains mandatory. A shared-slot shortcut would invalidate the comparison.

The source audit pins the CDB repository at `abdfae0adbdd12be8c6123ccb34eee98a38afb39`.
No CDB throughput figure serves as a PARPP measurement.

## Novelty limit

Prediction, stale activations, pipeline partitioning, and distributed execution each have prior work.
This experiment tests their joint feasibility under causal recurrent-stage inputs and free-running language generation.
Activation similarity alone does not establish a new architecture or a useful optimization.
Any architecture claim requires task quality and real concurrent execution against fair exact baselines.
