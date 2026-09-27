# Mixed-domain online vision-signature calibration PoC

This bounded PoC tests whether causally available, modality-separated visual/text prefill routing signatures improve prediction of later decode MoE/EP workload in a mixed-domain stream. It does not change routing or implement expert skipping.

The authoritative design is `poc_flashvep/reports/MLLM_Mixed_Domain_Online_Vision_Signature_Calibration_EP2_EP4_EP8_PoC.md` in the source workspace.

## Reproduction

From the repository root:

```bash
python -m poc_mllm_mixed_domain_online_calibration.scripts.build_cohort \
  --output poc_mllm_mixed_domain_online_calibration/artifacts/manifests
bash poc_mllm_mixed_domain_online_calibration/scripts/run_ep4_trace.sh
python -m poc_mllm_mixed_domain_online_calibration.scripts.consolidate_trace \
  --topology ep4 \
  --trace-dir poc_mllm_mixed_domain_online_calibration/artifacts/raw_traces/ep4_mixed160 \
  --results poc_mllm_mixed_domain_online_calibration/artifacts/runs/ep4_mixed160/results.jsonl \
  --output poc_mllm_mixed_domain_online_calibration/artifacts/ep4_compact.jsonl.gz \
  --audit poc_mllm_mixed_domain_online_calibration/artifacts/ep4_trace_audit.json
bash poc_mllm_mixed_domain_online_calibration/scripts/run_ep2_trace.sh
python -m poc_mllm_mixed_domain_online_calibration.scripts.consolidate_trace \
  --topology ep2 \
  --trace-dir poc_mllm_mixed_domain_online_calibration/artifacts/raw_traces/ep2_mixed48 \
  --results poc_mllm_mixed_domain_online_calibration/artifacts/runs/ep2_mixed48/results.jsonl \
  --output poc_mllm_mixed_domain_online_calibration/artifacts/ep2_compact.jsonl.gz \
  --audit poc_mllm_mixed_domain_online_calibration/artifacts/ep2_trace_audit.json
python -m poc_mllm_mixed_domain_online_calibration.scripts.analyze
python -m poc_mllm_mixed_domain_online_calibration.scripts.generate_reports
python -m pytest -q poc_mllm_mixed_domain_online_calibration/tests
```

The wrappers stop only the user's burner workers on measured GPUs and restore them through a shell trap. Raw GPU traces and copied images stay local; compact evidence and frozen manifests are versioned.

EP8 is structural replay only. No EP8 latency, TPOT, E2E, or speedup is emitted.
