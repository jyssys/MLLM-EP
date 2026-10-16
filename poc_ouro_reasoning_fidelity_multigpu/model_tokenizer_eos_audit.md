# Model, tokenizer, and EOS audit

The checkpoint revision is `f1edd81e7ac41355db670500ceaf204e0f73af68`.
The safetensors file contains 5,336,011,242 bytes.
Its SHA256 is `c506a79247dc51fc0400d789365c3d43932f718abce9810f3606ace47d0a3080`.
The model-source SHA256 is `263c097a2ed6870ce549a65ef6d6bb9e8e167423f87d2c494e0988b9a4f63f4b`.
The dynamic module matches that pinned source.

The loaded config specifies 48 physical layers, four passes, and 16 attention/KV heads.
The intended fixed-depth execution contains 192 logical layer executions.
The native runtime registers separate KV state for all 192 slots.
The card's 24-layer row does not match the actual checkpoint config.
The experiment does not change the checkpoint to match that row.

[Paper Table 2](https://arxiv.org/html/2510.25741v2#S4.T2) lists 48 layers for Ouro 2.6B.
The measured depth therefore agrees with that paper table, not the card row.

The tokenizer uses BOS `<|im_start|>` ID1 and EOS `<|im_end|>` ID2.
It has no configured pad token.
There is no separate generation_config.json in this snapshot.
The audit records the complete config and all relevant file hashes.

P0 exactly matches all 130 previous frozen input-ID sequences.
P1 changes only the assistant boundary by adding `<think>\n`.
The added IDs are `[3, 198]`.
The pinned template therefore implements the requested thinking option.
The model-card arithmetic example is a formatting diagnostic, not a third score-selected benchmark arm.
All rendered strings, escaped text, ID sequences, and hashes remain available.

| Arm | Assistant boundary bytes | Added thinking IDs | Pilot output opening |
|---|---|---|---|
| P0 | `<\|im_start\|>assistant\n` | None | ID3 in 8/8 outputs |
| P1 | `<\|im_start\|>assistant\n<think>\n` | `[3, 198]` | No duplicate opening ID3 in 8/8 outputs |

The table uses escaped newlines.
The full input bytes remain in `results/frozen_questions.json`.
The pilot output counts are descriptive, not an untouched benchmark comparison.

The natural native lanes keep `ignore_eos=False` and use no extra stop string.
The driver audit records the effective renderer EOS ID.
Native generated-token counts include the terminal EOS ID when it is present.
Decoded text omits that special token.

Teacher-forced state probes separately use `ignore_eos=True` to preserve their exact prefixes.
Those probes cannot supply accuracy or natural completion rates.
The HF reference explicitly selects final pass index3, which means R4.
It does not use a shallower recurrent state as the quality reference.

## Reference-version failure

The card recommends Transformers 4.54.1.
The current pinned cache constructor assigns `self.key_cache`.
Transformers 4.54.1 exposes that name as a read-only property.
The official-code constructor therefore raises AttributeError before cached generation.
The same pinned constructor works under the existing Transformers 4.57.1 environment.
The CPU regression fills and checks all 192 cache slots there.

This is a measured code/API incompatibility for this pinned combination.
It is not evidence of weak pretrained reasoning quality.
No shared environment or model-source file changes.
The functioning reference retains an explicit version-drift limitation.
The failed physical run and both CPU constructor results remain preserved.

## Cross-version token and text checks

The isolated 4.54.1 and 4.57.1 tokenizers each encode all 260 frozen prompt arms identically to the stored IDs.
These arms comprise 130 questions and two prompt formats.

Both tokenizers also decode 92 captured output sequences identically to native text.
The sequences comprise the complete 90-output AIME 8,192-cap lane and two MATH continuity outputs.
The unchanged final-answer parser returns identical extractions in both comparisons.
No primary score changes because of this audit.

The native 5.16.1 encoder has four input-arm differences when it re-encodes text.
Every model lane receives frozen IDs directly, so those re-encoding differences do not enter the experiment.
These observations do not establish universal tokenizer equivalence or private-grader equivalence.

Evidence: `results/output_text_audit_4.54.1.json`, `results/output_text_audit_4.57.1.json`, and `results/native_tokenizer_preflight.json`.

## Sampling mask warning

The unpadded HF B1 sampling probe uses pad=eos=2 for generation bookkeeping.
The library warns that it cannot infer padding from that shared ID.
The installed fallback returns an all-ones input mask in this case.
A direct CPU call confirms that behavior on both frozen MATH inputs.
It does not mask the ChatML EOS markers inside the prompt.
This audit does not replace numerical or benchmark-quality validation.

## Arithmetic completion diagnostic

The official-card arithmetic probe generates 562 tokens and ends with EOS.
It closes the thinking section and states `x = 4` in the final response.
It retains 192 cache slots.

The frozen benchmark parser returns UNPARSEABLE for its unboxed `**Answer**` format.
The raw `parsed_correct=False` field records that parser result.
It does not mean the displayed arithmetic solution is wrong.
This manual format inspection changes no benchmark answer or parser rule.

Primary evidence is in `results/model_tokenizer_audit.json`, `results/cache_api_*.json`, and `results/HF_sampling_mask_audit.json`.
The final report states the remaining runtime and quality gates separately.
