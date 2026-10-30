"""CPU tests and source accounting for the existing mixed continuation path."""
import hashlib
import subprocess
from .common import ROOT, REPO, RLT, CF_COMMIT, OWNED, setup, write, digest


def main():
    setup()
    import sys
    sys.path.insert(0, str(RLT))
    from vllm_rlt.request import Request, RequestPhase, Operation
    from vllm_rlt import SamplingParams, SchedulerConfig
    from vllm_rlt.core.scheduler import ScheduledRequest, SchedulerOutput
    from vllm_rlt.core.scheduling_policy import MixedPolicy
    from vllm_rlt.worker.executor import build_execution_batch

    results = []
    for depth, cut in ((4, 2), (32, 16)):
        decode = Request('decode', [1]*128, SamplingParams(min_loops=depth, max_loops=depth),
                         generated_token_ids=[2], loops_done=0, phase=RequestPhase.RECURRENT)
        prompt = Request('continuation', [3]*128, SamplingParams(min_loops=depth, max_loops=depth),
                         loops_done=cut, phase=RequestPhase.PREFILLING_RECURRENT)
        scheduled = SchedulerOutput(None, [ScheduledRequest(decode, depth=0, phase=RequestPhase.RECURRENT),
                                           ScheduledRequest(prompt, token_count=128, depth=cut,
                                                            chunk_start=0, phase=RequestPhase.PREFILLING_RECURRENT)], 1)
        batch = build_execution_batch(scheduled, depth)
        assert batch.operation == Operation.RECURRENT and batch.num_rows == 129
        assert [span.depth for span in batch.spans] == [0, cut]
        assert [span.emit_gate for span in batch.spans] == [True, False]

        class Ready:
            def has_ready(self, phase):
                return phase in (RequestPhase.RECURRENT, RequestPhase.PREFILLING_RECURRENT)
            def take_phases(self, phases):
                assert phases == (RequestPhase.RECURRENT, RequestPhase.PREFILLING_RECURRENT)
                return scheduled

        policy = MixedPolicy(SchedulerConfig(mode='mixed'))
        assert policy.schedule(Ready()) is scheduled
        results.append(dict(depth=depth, cut=cut, combined_rows=batch.num_rows,
                            distinct_depths=[span.depth for span in batch.spans], passed=True))
    paths = [REPO/'poc_flashvep/looped_pd_abc'/name for name in ('split.py', 'worker.py', 'boundary.py', 'verify.py')]
    paths += [RLT/'vllm_rlt'/name for name in ('core/scheduler.py', 'core/scheduling_policy.py',
                                               'worker/executor.py', 'worker/model_runner.py',
                                               'attention/metadata.py', 'kernels/flash_attention.py',
                                               'models/ouro.py', 'models/huginn.py', 'pd/connector.py')]
    index = subprocess.check_output(['git', 'ls-files', '--stage'], cwd=REPO)
    kept = b''.join(line for line in index.splitlines(keepends=True)
                    if not any(line.split(b'\t',1)[-1].decode().strip().startswith(prefix) for prefix in OWNED))
    index_hash = hashlib.sha256(kept).hexdigest()
    write(ROOT/'configs/cpu_audit.json', dict(correctness_commit=CF_COMMIT, tests=results,
                                               source_hashes={str(p):digest(p) for p in paths},
                                               initial_git_index_sha256=index_hash,
                                               original_feature='A plus Native Mixed already accepts distinct recurrent depths in one physical operation',
                                               evidence='CPU_SOURCE_AND_UNIT_TEST',
                                               measured_GPU_speedup='NOT_MEASURED'))
    print(results)


if __name__ == '__main__':
    main()
