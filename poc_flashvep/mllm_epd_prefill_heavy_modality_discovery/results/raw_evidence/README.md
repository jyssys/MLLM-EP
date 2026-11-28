# Native EPD evidence archives

Each epoch has a compressed input tree for the offline reducer and a manifest with SHA-256 checksums. Per-file receipts record the original byte count and digest, the exported digest where applicable, and how the exporter handled the file. Epoch completion and individual cohort completion are recorded separately. A snapshot of a live epoch covers the recorded byte prefix of each file. A partial JSONL record is omitted and its bytes remain covered by the original-source digest.

The archives retain the cohort and arrival contracts, hashed prompt identity, generated token IDs and timings, stage and queue events, measured transfer counters and sizes, and all recorded GPU kernel and copy intervals with clock anchors. Full prompts, input token IDs, model tensors, media, literal response text, and CUDA IPC handles are excluded. The roster files carry geometry and identity for the reducer, and the original media and source manifests remain external for fresh serving runs. Environment, state, and log files outside the evidence selection are recorded by digest where captured in the listed sections.

Run the offline reducer from the checkout root with Python 3.11 or newer.

```bash
study=poc_flashvep/mllm_epd_prefill_heavy_modality_discovery
mod=poc_flashvep.mllm_epd_prefill_heavy_modality_discovery
evidence_root=$(mktemp -d)
for bundle in "$study"/results/raw_evidence/*.tar.gz; do
    sha256sum "$bundle"
    tar -xzf "$bundle" -C "$evidence_root"
done
python -m "$mod.reduce_epd" --raw-root "$evidence_root" --out "$evidence_root/reduced"
```

Compare each checksum to `archive_sha256` in the matching manifest. The reduced request-file receipt identifies the sanitized input file used for replay. The evidence manifest maps that receipt to the original raw bytes. Generated timing, queue, counter, and interval summaries can be reproduced from the curated files on a host without GPUs.

Connector READ observations preserve host timestamps, outcome and count fields. A SHA-256 digest of the producer engine and request identity joins each original P/H descriptor to D without retaining either opaque identifier.

Client rows preserve the SHA-256 of canonical input token IDs. This supports exact cross-path input checks while keeping full input token IDs out of the published evidence.

The historical AR comparison can be reproduced from the already published native trace. Extract the pinned Qwen3-8B case and pass it to the same reducer. The reducer checks both worker CSV hashes against that prior report's published receipts.

```bash
prior=5bdd6ef5e91840e37c776c2ad937866531e6a391
ar_archive=poc_flashvep/pd_resource_complementary_parallelism/results/raw_native.tar.gz
git show "$prior:$ar_archive" | tar -xzf - -C "$evidence_root" --wildcards 'dense-controls/p-light/*'
python -m "$mod.reduce_epd" --raw-root "$evidence_root" --out "$evidence_root/reduced" \
    --ar-profile "$evidence_root/dense-controls/p-light/16k_light_profile"
```

To refresh a local bundle from the original campaign, run the export command below with the correct raw root. Omit `--epoch` to export all campaign run directories. The exporter checks each existing archive and skips unchanged snapshots.

```bash
python -m "$mod.archive" --raw-root ../h200_epd_modality_runs \
    --out "$study/results/raw_evidence" --epoch calibration-sweep
```
