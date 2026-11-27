#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo 'Usage: run_online_case.sh CASE UNIQUE_RUN_NAME [native_study options]' >&2
    exit 2
fi
case_name=$1
run_name=$2
shift 2
if [[ ! "$run_name" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo 'Use a unique simple run name.' >&2
    exit 2
fi
study_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${study_dir}/../.."
study_runs=${MLLM_MEMORY_RUNS:-../h200_memory_asymmetry_runs}
study_python=${MLLM_MEMORY_PYTHON:-../mllm_epd_runtime_env/bin/python}
study_asset=${MLLM_MEMORY_ASSET:-../MLLM-EP-mllm-epd}
study_receipt=${MLLM_MEMORY_RECEIPT:-../mllm_epd_autosetup/download_ready_1795600931728841878.json}

case "$case_name" in
    native-natural)
        manifest=natural149_manifest
        options=(--qps 2.5 --cycles 2)
        ;;
    encoder-natural)
        manifest=natural149_manifest
        options=(--stable --encoder-cache --qps 2.5 --cycles 2)
        ;;
    encoder-host-natural)
        manifest=natural149_manifest
        options=(--stable --encoder-cache --encoder-cache-device cpu --qps 2.5 --cycles 2)
        ;;
    combined-host-embedding-natural)
        manifest=natural149_manifest
        options=(--stable --encoder-cache --encoder-cache-device cpu --apc-roles prefill decode --qps 2.5 --cycles 2)
        ;;
    apc-natural)
        manifest=natural149_manifest
        options=(--stable --apc-roles prefill decode --qps 2.5 --cycles 2)
        ;;
    combined-natural)
        manifest=natural149_manifest
        options=(--stable --encoder-cache --apc-roles prefill decode --qps 2.5 --cycles 2)
        ;;
    peer-natural)
        manifest=natural149_manifest
        options=(--stable --encoder-cache --apc-roles prefill decode --tier peer --tier-gib 16 --qps 2.5 --cycles 2)
        ;;
    pressure-native)
        manifest=w5_eviction188_manifest
        options=(--stable --encoder-cache --apc-roles prefill decode --qps 2.5 --output 16)
        ;;
    pressure-local)
        manifest=w5_eviction188_manifest
        options=(--stable --encoder-cache --apc-roles prefill decode --p-kv-gib 72 --qps 2.5 --output 16)
        ;;
    pressure-peer)
        manifest=w5_eviction188_manifest
        options=(--stable --encoder-cache --apc-roles prefill decode --tier peer --tier-gib 64 --qps 2.5 --output 16)
        ;;
    pressure-peer16-56)
        manifest=w5_eviction188_manifest
        options=(--stable --encoder-cache --apc-roles prefill decode --tier peer --tier-gib 56 --p-kv-gib 16 --qps 2.5 --output 16)
        ;;
    pressure-peer-matched)
        manifest=w5_eviction188_manifest
        options=(--stable --encoder-cache --apc-roles prefill decode --tier peer --tier-gib 64 --p-kv-gib 8 --qps 2.5 --output 16)
        ;;
    pressure-host-matched)
        manifest=w5_eviction188_manifest
        options=(--stable --encoder-cache --apc-roles prefill decode --tier cpu --tier-gib 64 --p-kv-gib 8 --qps 2.5 --output 16)
        ;;
    pressure-host)
        manifest=w5_eviction188_manifest
        options=(--stable --encoder-cache --apc-roles prefill decode --tier cpu --tier-gib 64 --qps 2.5 --output 16)
        ;;
    unique-native)
        manifest=unique_images_manifest
        options=(--qps 2.5)
        ;;
    unique-peer)
        manifest=unique_images_manifest
        options=(--stable --encoder-cache --apc-roles prefill decode --tier peer --tier-gib 16 --qps 2.5)
        ;;
    *)
        echo "Unknown case: $case_name" >&2
        exit 2
        ;;
esac
export PYTHONDONTWRITEBYTECODE=1
exec "$study_python" -u -m poc_flashvep.mllm_epd_memory_asymmetry_discovery.native_study \
    --asset-repo "$study_asset" --receipt "$study_receipt" \
    --manifest "$study_runs/$manifest.json" --run-dir "$study_runs/$run_name" "${options[@]}" "$@"
