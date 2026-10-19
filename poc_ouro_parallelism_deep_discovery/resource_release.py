"""Prerequisite release for campaign-owned sequential GPU queues."""
from poc_ouro_parallelism_deep_discovery.contract import read


def release_complete(directory):
    status = directory / 'guard_status.json'
    cleanup = directory / 'guard_cleanup_validation.json'
    if not status.exists() or not cleanup.exists():
        return False
    return bool(read(status).get('success')) and read(cleanup).get('owned_GPU_contexts_remaining') == []
