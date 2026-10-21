"""Freeze observed Single-Dynamic token/depth prefixes. No fabricated exits."""
from .contract import *


def main():
    source=ROOT/'raw/quality_SD_pilot_v1';requests=[];files=[]
    for i in range(16):
        file=source/f'gsm8k_test_{i}.json';r=read(file)
        assert r['arm']=='SD' and r['depths'][0]==4 and r['prompt_ids']==dataset(True)[i]['input_ids']
        requests.append(dict(id=f'depth-replay-{i}',dataset='frozen_hardware',source_id=r['id'],input_ids=r['prompt_ids'],
            teacher_ids=r['output_ids'],teacher_depths=r['depths'],prompt_sha256=r['prompt_sha256'],stop_strings=[]))
        files.append(dict(path=str(file.relative_to(ROOT)),sha256=sha(file)))
    cap=min(32,min(len(r['teacher_ids']) for r in requests))
    for r in requests:r['teacher_ids']=r['teacher_ids'][:cap];r['teacher_depths']=r['teacher_depths'][:cap]
    for C in [1,4,8,16]:
        path=ROOT/f'dynamic_replay_C{C}.json';assert not path.exists()
        write(path,dict(name=f'Live_SD_prefix_C{C}',C=C,O=cap,requests=requests[:C],sources=files[:C],
            rule='First canonical GSM requests. O=min(32, shortest recorded generated output). No latency or accuracy selection.',
            warning='Future exit depths are trace-locked hardware replay, not live gate decisions or a deployable policy.'))
    print('FROZEN_DYNAMIC_REPLAY',cap,flush=True)


if __name__=='__main__':main()
