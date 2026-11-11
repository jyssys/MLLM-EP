"""A stopped receipt does not establish a successful worker process exit."""
import os
from .common import ROOT, write


def validate_exit(code):
    if code != 0:
        raise RuntimeError(f'Native worker did not exit successfully: {code}')


def install(cfg):
    from vllm_rlt.engine.llm_engine import StageClient
    original = StageClient.join
    records = []

    def joined(client, *args, **kwargs):
        result = original(client, *args, **kwargs)
        code = client.process.exitcode
        records.append(dict(stage=client.stage_config.stage_id,
            role=client.stage_config.role, pid=client.process.pid, exitcode=code,
            forced_cleanup=bool(kwargs.get('force'))))
        write(ROOT/'raw'/cfg['name']/f'native_worker_exit_{os.getpid()}.json', dict(
            records=records, passed=all(row['exitcode'] == 0 for row in records),
            scope='Actual worker process exit after Native join. Output completion and a stopped receipt are separate conditions.'))
        # A fatal deployment must still terminate every receiver after its
        # writers. Do not interrupt Native forced-cleanup ordering.
        if not kwargs.get('force'):
            validate_exit(code)
        return result
    StageClient.join = joined
