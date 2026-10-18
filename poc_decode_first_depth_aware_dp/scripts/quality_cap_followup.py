"""Run the predefined 8192-token paired follow-up. Preserve the primary cap analysis."""
import concurrent.futures
import datetime
import os
import subprocess
from poc_decode_first_depth_aware_dp.contract import *


def main():
    manifest=ROOT/'artifacts/quality_cap_followup_controller.json'
    assert not manifest.exists()
    assert not (ROOT/'artifacts/quality_cap_followup_frozen.json').exists()
    primary=ROOT/'artifacts/quality_pilot_analysis_v2/quality_predictions.json'
    rows=read(primary)
    ids=sorted({r['id'] for r in rows if r['truncated']})
    write(ROOT/'artifacts/quality_cap_followup_frozen.json',dict(ids=ids,
        frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        criterion='Union of truncated primary question IDs. Rerun both policies for every selected ID.',
        before_followup_inference=True,primary_predictions_sha256=sha(primary),
        cap=8192,max_model_len=10240,physical_KV_token_capacity=640*16,
        policy_changes=False,primary_analysis_replaced=False,
        status='PENDING' if ids else 'NOT_NEEDED'))
    if not ids:
        write(manifest,dict(all_success=True,assignments=[],selected_ids=[],scope='No truncated primary output.'))
        return
    def run(gpu,policy,part):
        command=[REFERENCE_PYTHON,'-m','poc_decode_first_depth_aware_dp.scripts.gpu_session',
            '--gpus',str(gpu),'--output',f'outputs/quality_{policy}_part{part}_cap8192_v1',
            '--timeout','7200','--',PYTHON,'-m','poc_decode_first_depth_aware_dp.scripts.quality_run',
            '--policy',policy,'--part',str(part),'--protocol',str(ROOT/'artifacts/quality_protocol_v3_pilot32.json'),
            '--max-new-tokens','8192','--ids-only',*ids]
        result=subprocess.run(command,cwd=PROJECT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=MASK))
        return dict(gpu=gpu,policy=policy,part=part,command=command,return_code=result.returncode)
    assignments=[(4,'fixed',0),(5,'fixed',1),(6,'dynamic',0),(7,'dynamic',1)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda item:run(*item),assignments))
    write(manifest,dict(assignments=results,all_success=all(r['return_code']==0 for r in results),
        selected_ids=ids,scope='Paired cap sensitivity only. Does not replace the common 4096-token primary analysis.'))
    assert all(r['return_code']==0 for r in results)


if __name__=='__main__':main()
