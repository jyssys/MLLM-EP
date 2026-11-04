"""Run a collector self-test through the existing resource guard."""
from .common import *
from .recovery_qualification import save
from .run import execute


def main():
    initialize()
    cfg = json.loads((PREVIOUS / 'configs/collector_activity_toggle_selftest.json').read_text())
    cfg.update(name='activity_only_collector_selftest', worker='collector_selftest',
               activity_selftest=True, timeout=120, estimated_seconds=30,
               gpus=[6, 7], phase='INSTRUMENTATION_ONLY_NOT_MODEL_BENCHMARK')
    save(cfg)
    assert execute(cfg['name'])


if __name__ == '__main__':
    main()
