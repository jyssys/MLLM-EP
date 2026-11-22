"""Complete CPU delivery after the verified archive, even if chat disconnects."""
import json
import os
from pathlib import Path
import subprocess
import time
from .common import ROOT,REPO,PYTHON,BRANCH,write,digest,preserve,resources


def alive(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False


def main():
    target=ROOT/'archives/evidence_archive_manifest_v1.json'
    assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists()
    controller=json.loads((ROOT/'state/background_final_lossless_archive_v2.json').read_text())['pid']
    while not target.exists():
        if not alive(controller):
            assert target.exists(),'Archive controller stopped before its verified manifest'
            break
        time.sleep(10)
    assert json.loads(target.read_text())['verified_lossless']
    # Its final stdout is part of the evidence. Wait until that log is stable.
    while alive(controller):time.sleep(1)
    preserve()
    subprocess.run([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_longbench400_pd.final_work_accounting'],cwd=REPO,check=True)
    accounting=json.loads((ROOT/'state/final_work_accounting.json').read_text())
    assert accounting['all_required_work_accounted'] and not accounting['unresolved_mandatory_work']
    subprocess.run([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_longbench400_pd.publication','--push'],cwd=REPO,check=True)
    receipt=json.loads((ROOT/'state/publication_receipt.json').read_text())
    assert receipt['pushed'] and receipt['remote_matches']
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=REPO,text=True).split()[0]
    assert remote==receipt['SHA']==receipt['Remote_SHA']
    owner=resources();samples=[]
    script=Path('/home/esjung/vllm-ep/run_utilize.sh')
    original=json.loads((ROOT/'state/environment_initial.json').read_text())
    assert digest(script)==original['approved_script_SHA256']
    for _ in range(30):
        text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used',
            '--format=csv,noheader,nounits'],text=True)
        values=[dict(GPU=int(a),UUID=b,utilization_percent=int(c),memory_MiB=int(d))
            for a,b,c,d in (line.split(', ') for line in text.strip().splitlines())]
        active=owner.processes();burners=owner.burners()
        owned=all(os.getsid(r['pid'])==burners.get(r['gpu']) for r in active)
        ready=owned and {r['gpu'] for r in active}==set(range(8)) and all(r['utilization_percent']>=70 for r in values)
        samples.append(dict(time=time.time(),GPUs=values))
        if ready:break
        time.sleep(2)
    assert ready,'Keep the pushed evidence. Inspect resource ownership without stopping an unknown process.'
    preserve()
    write(ROOT/'state/post_publication_final_check.json',dict(time=time.time(),passed=True,
        branch=BRANCH,SHA=receipt['SHA'],Remote_SHA=remote,remote_matches=True,
        verified_archive_manifest_SHA256=digest(target),all8_approved_burners_running=True,
        active_processes=active,burner_sessions=burners,utilization_samples=samples,
        user_checkout_and_index_preserved=True,
        scope='Post-commit local receipt. No extra GPU experiment or process termination occurs. '
              'Keep the fresh approved all-eight burners running after this check. '
              'This operational success does not convert scientific HOLD or NO-GO into GO.'))
    print('FINAL_DELIVERY_COMPLETE',BRANCH,remote,flush=True)


if __name__=='__main__':main()
