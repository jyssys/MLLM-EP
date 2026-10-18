"""Run the next frozen quality questions. These workers are not a DP benchmark."""
import concurrent.futures
import os
import subprocess
from poc_decode_first_depth_aware_dp.contract import *


def main():
    manifest=ROOT/'artifacts/quality_extension_controller.json'
    assert not manifest.exists()
    protocol=ROOT/'artifacts/quality_protocol_v3_pilot32.json'
    assert len(read(protocol)['extension_ids'])==32
    def run(gpu,policy,part):
        command=[REFERENCE_PYTHON,'-m','poc_decode_first_depth_aware_dp.scripts.gpu_session',
            '--gpus',str(gpu),'--output',f'outputs/quality_{policy}_part{part}_extension_v1',
            '--timeout','7200','--',PYTHON,'-m','poc_decode_first_depth_aware_dp.scripts.quality_run',
            '--policy',policy,'--part',str(part),'--protocol',str(protocol),'--extension-only']
        result=subprocess.run(command,cwd=PROJECT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=MASK))
        return dict(gpu=gpu,policy=policy,part=part,command=command,return_code=result.returncode)
    assignments=[(4,'fixed',0),(5,'fixed',1),(6,'dynamic',0),(7,'dynamic',1)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda item:run(*item),assignments))
    write(manifest,dict(assignments=results,all_success=all(r['return_code']==0 for r in results),
        scope='Stage-0 pilot extension only. No live DP placement or scheduling experiment.'))
    assert all(r['return_code']==0 for r in results)


if __name__=='__main__':main()
