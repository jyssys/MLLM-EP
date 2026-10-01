# Reproduction

Run from the repository root with the validated environment. Stop only the
owner's burner on GPUs 2,3,6,7 before physical timing.

```bash
export CUDA_VISIBLE_DEVICES=2,3,6,7
export PYTHONPATH="$PWD/poc_mllm_ep_static4_comm_only/hooks:$PWD"
export MLLM_STATIC4_ENABLE=1
export MLLM_STATIC4_OUTPUT="$PWD/poc_mllm_ep_static4_comm_only/artifacts/static4_layer_replay/repro"
export MLLM_COMM_OUTPUT="$MLLM_STATIC4_OUTPUT"
export MLLM_COMM_CONTROL="$PWD/poc_mllm_ep_static4_comm_only/artifacts/control_repro.json"

/home/esjung/.venvs/flashvep-deepep-v020/bin/python \
  poc_mllm_ep_static4_comm_only/scripts/run_static4_serving.py \
  --manifest poc_mllm_ep_communication_motivation/artifacts/manifests/frozen_manifest.json \
  --output "$MLLM_STATIC4_OUTPUT" --control "$MLLM_COMM_CONTROL" \
  --mode correctness --repeats 3 --decode-tokens 32 --injection-progress 16
```

For timing, use a new output directory, `--mode timing --repeats 2
--clean-native`, and repeat five independent sessions. This is intentional:
one 10-repeat external-DP process eventually hung during teardown, whereas the
five bounded sessions completed and preserve ten paired request units.

Analysis and tests:

```bash
PYTHONPATH=$PWD /home/esjung/.venvs/flashvep-deepep-v020/bin/python \
  poc_mllm_ep_static4_comm_only/scripts/analyze_static4.py \
  --project poc_mllm_ep_static4_comm_only

PYTHONPATH=$PWD /home/esjung/.venvs/flashvep-deepep-v020/bin/python \
  -m unittest discover -s poc_mllm_ep_static4_comm_only/tests -v
ruff check poc_mllm_ep_static4_comm_only
```

The large raw JSONL streams are intentionally git-ignored. Compact measured
statistics and verdicts are under `artifacts/results/` and
`artifacts/final_verdict.json`.
