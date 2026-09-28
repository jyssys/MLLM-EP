# Vision-dominant captioning EP evidence PoC

This directory is an isolated, bounded evidence study. It preserves the frozen Qwen3-VL
BF16 K8 route and does not implement routing, skipping, placement, or serving optimization.

The cohort uses one byte-identical prompt across 200 unique images (five balanced domains).
DOCCI, TextCaps, and NoCaps use public caption assets. The dense-relation and scene/layout
lanes use the documented MMStar substitutes because paragraph-caption/ADE20K pixels were not
available locally. This deviation is frozen before router or generated-output inspection.

Rebuild the manifest:

```bash
PYTHONPATH=$PWD /home/esjung/.venvs/libra-supplement-py310/bin/python \
  -m poc_mllm_vision_dominant_captioning.scripts.acquire_public_sources
PYTHONPATH=$PWD /home/esjung/.venvs/libra-supplement-py310/bin/python \
  -m poc_mllm_vision_dominant_captioning.scripts.build_cohort
```

Instrumented EP4 natural-generation command used for the accepted v2 trace:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 MLLM_VDC_TRACE=1 \
PYTHONPATH=$PWD/poc_mllm_vision_dominant_captioning/hooks:$PWD \
/home/esjung/.venvs/flashvep-deepep-v020/bin/python \
  -m poc_mllm_vision_dominant_captioning.scripts.run_physical \
  --topology ep4 --physical-gpus 4,5,6,7 --trace --max-tokens 256 \
  --pool poc_mllm_vision_dominant_captioning/artifacts/manifests/frozen_image_cohort.json \
  --stream poc_mllm_vision_dominant_captioning/artifacts/manifests/frozen_iid_stream.json \
  --output poc_mllm_vision_dominant_captioning/runs/ep4_natural200_v2 \
  --trace-dir poc_mllm_vision_dominant_captioning/artifacts/raw/ep4_natural200_v2_trace \
  --control poc_mllm_vision_dominant_captioning/artifacts/ep4_natural200_v2_control.json \
  --ledger poc_mllm_vision_dominant_captioning/artifacts/execution_ledger.jsonl
```

Reproduce the frozen offline analysis and reports from the checked-in compact evidence:

```bash
PYTHONPATH=$PWD /home/esjung/.venvs/flashvep-deepep-v020/bin/python \
  -m poc_mllm_vision_dominant_captioning.scripts.analyze_caption \
  --ep4-compact poc_mllm_vision_dominant_captioning/artifacts/ep4_compact.jsonl.gz \
  --natural-results poc_mllm_vision_dominant_captioning/artifacts/ep4_natural_metadata.jsonl.gz \
  --ep2-compact poc_mllm_vision_dominant_captioning/artifacts/ep2_compact.jsonl.gz \
  --causal-compact poc_mllm_vision_dominant_captioning/artifacts/ep4_causal_compact.jsonl.gz \
  --ep2-causal-compact poc_mllm_vision_dominant_captioning/artifacts/ep2_causal_compact.jsonl.gz

PYTHONPATH=$PWD /home/esjung/.venvs/flashvep-deepep-v020/bin/python \
  -m poc_mllm_vision_dominant_captioning.scripts.render_reports \
  --natural-results poc_mllm_vision_dominant_captioning/artifacts/ep4_natural_metadata.jsonl.gz
```

The large raw trace and image pixels are intentionally ignored. Compressed routing summaries,
frozen manifests, result metadata, audits, tests, reports, and the evidence verdict are checked in.
The committed metadata export under `artifacts/ep4_natural_metadata.jsonl.gz` is sufficient for
offline regeneration; the reports are the authoritative result.

Use `scripts/run_guarded.sh` for future runs so the existing GPU4-7 burner policy is
restored by a shell trap on success, error, or signal.
