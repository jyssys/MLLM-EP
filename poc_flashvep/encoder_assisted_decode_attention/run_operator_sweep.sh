#!/usr/bin/env bash
set -euo pipefail

eada_dir=$1
eada_python=$2
eada_model=$3
eada_manifest=$4
eada_module=poc_flashvep.encoder_assisted_decode_attention.operator_bench
eada_root=$(dirname "$(dirname "$eada_python")")
export CUDA_HOME="$eada_root" PATH="$eada_root/bin:$PATH"
export CC="$eada_root/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$eada_root/bin/x86_64-conda-linux-gnu-g++"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$eada_dir"

eada_run() {
    "$eada_python" -u -m "$eada_module" --decode-device 2 --helper-device 0 --producer-device 1 "$@"
}

for eada_candidate in memcpy peer_kernel; do
    if eada_run --out "$eada_dir/smoke_$eada_candidate.jsonl" --contexts 32768 --batch 8 32 \
            --ratios .5 --require-split-graph --graph-transport "$eada_candidate" \
            --calls 24 --repeats 5 --link-repeats 128 >"$eada_dir/smoke_$eada_candidate.log" 2>&1; then
        touch "$eada_dir/smoke_$eada_candidate.complete"
    fi
done

eada_transport=$("$eada_python" - "$eada_dir" <<'PY'
import json
from pathlib import Path
from statistics import median
import sys

folder = Path(sys.argv[1])
results = {}
for name in ("memcpy", "peer_kernel"):
    if (folder / f"smoke_{name}.complete").exists():
        with (folder / f"smoke_{name}.jsonl").open() as fp:
            cases = [row for row in map(json.loads, fp) if row["type"] == "case"]
        results[name] = median([
            median(s["cuda_us"] for s in row["timings"]["split_graph"])
            / median(s["cuda_us"] for s in row["timings"]["local_best_graph"])
            for case in cases for row in case["outcomes"]
        ])
selected = min(results, key=results.get) if results else "eager"
with (folder / "transport_selection.json").open("x") as fp:
    json.dump({"smoke_graph_over_local": results, "selected": selected}, fp, indent=2)
    fp.write("\n")
print(selected)
PY
)

if [[ "$eada_transport" == eager ]]; then
    eada_graph_args=(--no-split-graph)
else
    eada_graph_args=(--graph-transport "$eada_transport")
fi
printf 'EADA selected operator transport %s\n' "$eada_transport"
eada_run --out "$eada_dir/controlled_paged.jsonl" "${eada_graph_args[@]}" \
    --contexts 4096 8192 16384 32768 65536 --batch 1 4 8 16 32 --ratios .25 .5 .75 1 \
    --repeats 4 --calls 20

eada_run --out "$eada_dir/native_qwen_ragged.jsonl" "${eada_graph_args[@]}" --model "$eada_model" \
    --manifest "$eada_manifest" --id lmms-lab__DocVQA__56399 --id yubo2333__MMLongBench-Doc__658 \
    --id lmms-lab__MP-DocVQA__52541 --id lmms-lab__MP-DocVQA__6771 \
    --repeats 6 --calls 24 --link-repeats 128
