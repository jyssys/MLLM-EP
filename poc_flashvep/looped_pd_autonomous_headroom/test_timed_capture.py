"""Check bounded-capture control without a GPU or a model substitute."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import patch
from . import timed_capture


def main():
    calls = []
    clock = [100.]
    class Engine:
        stage_config = SimpleNamespace(role='prefill')
        model_runner = SimpleNamespace(device='cuda:0')
        def _step(self):
            calls.append('native_step')
        def close(self):
            calls.append('native_close')
    native = ModuleType('vllm_rlt.engine.stage_engine')
    native.StageEngine = Engine
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(
        cudart=lambda: SimpleNamespace(cudaProfilerStart=lambda: calls.append('start'),
                                      cudaProfilerStop=lambda: calls.append('stop')),
        nvtx=SimpleNamespace(mark=lambda label: calls.append(label))))
    with tempfile.TemporaryDirectory(prefix='bounded_capture_test_') as temporary:
        root = Path(temporary)
        folder = root/'raw'/'test'
        folder.mkdir(parents=True)
        (folder/'measurement_begin.json').write_text(json.dumps(dict(time=100.)))
        cfg = dict(name='test', arrival_offsets=[0., 100.], autonomous_capture_seconds=3.)
        with patch.dict('sys.modules', {'torch': fake_torch, 'vllm_rlt.engine.stage_engine': native}), \
             patch.object(timed_capture, 'ROOT', root), \
             patch.object(timed_capture.time, 'monotonic', lambda: clock[0]):
            timed_capture.install(cfg)
            engine = Engine()
            engine._step()
            assert 'start' not in calls
            clock[0] = 135.
            engine._step()
            clock[0] = 137.
            engine._step()
            assert 'stop' not in calls
            clock[0] = 138.
            engine._step()
            engine.close()
            assert calls.count('start') == calls.count('stop') == 1
            assert calls.count('native_step') == 4
            events = [json.loads(row) for row in (folder/'capture_events.jsonl').read_text().splitlines()]
            assert [row['event'] for row in events] == ['START', 'STOP']
            assert events[0]['time'] == 135. and events[1]['stop_begin'] == 138.
    print('PASS: capture timing, one start/stop, native step preservation. CPU control test only.')


if __name__ == '__main__':
    main()
