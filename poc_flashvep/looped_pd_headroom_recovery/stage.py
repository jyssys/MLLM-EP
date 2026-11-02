"""Run the unchanged native StageWorker with one inherited IPC descriptor."""
import pickle
import sys
from pathlib import Path
from multiprocessing.connection import Connection


def main():
    import os,time
    from .common import write
    from .runtime import install
    install()
    from vllm_rlt.engine.worker import worker_main
    stage_config, device, model_source, options, kwargs = pickle.loads(Path(sys.argv[1]).read_bytes())
    write(Path(sys.argv[1]).parent/f'stage_owner_{stage_config.role}.json',dict(pid=os.getpid(),sid=os.getsid(0),uid=os.getuid(),
        time=time.time(),payload=str(Path(sys.argv[1]).resolve()),role=stage_config.role))
    worker_main(stage_config, device, Connection(int(sys.argv[2])), model_source, options, **kwargs)


if __name__ == '__main__':
    main()
