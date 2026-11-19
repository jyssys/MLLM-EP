"""Preserve the redirection record and verify the installed wheel sources."""
import base64
import collections
import hashlib
import importlib.metadata
import json
import time
from .common import ROOT,PREVIOUS,write,digest,initialize


def main():
    initialize();copied={}
    for path in sorted((PREVIOUS/'state').glob('research_redirection_*.json')):
        target=ROOT/'state'/path.name
        value=json.loads(path.read_text());write(target,value)
        copied[str(path)]=dict(SHA256=digest(path),copy=str(target),copy_SHA256=digest(target))
        assert digest(path)==digest(target)
    freeze=json.loads((PREVIOUS/'state/research_redirection_pre_stop_snapshot.json').read_text())
    inventory=[]
    for path in sorted((PREVIOUS/'configs').glob('*.json')):
        cfg=json.loads(path.read_text())
        if not isinstance(cfg,dict) or cfg.get('name')!=path.stem:continue
        folder=PREVIOUS/'raw'/cfg['name'];receipt=folder/'dispatcher_result.json'
        if cfg['name']=='sweep_Token_L16384_Last4_N36_O16_q8_B1_v1':
            status='STOPPED_BY_RESEARCH_REDIRECTION';scope='Excluded. Unexpected dispatch after intercepted SIGTERM. Coordinator interrupted and workers cleaned up.'
        elif cfg['name']==freeze['active_lease']['name']:
            status='STOPPED_BY_RESEARCH_REDIRECTION';scope='The in-flight child finished normally. The controller did not resume further authorized research.'
        elif receipt.exists():
            r=json.loads(receipt.read_text());status='COMPLETED_BEFORE_REDIRECTION' if r['complete'] else 'PREEXISTING_FAILED'
            scope='Original result status remains unchanged.'
        else:
            status='NOT_RUN';scope='No new execution after research redirection.'
        inventory.append(dict(name=cfg['name'],status=status,scope=scope,config_SHA256=digest(path),
            raw=str(folder),source_archive_SHA256=digest(folder/'source.zip') if (folder/'source.zip').exists() else None))
    write(ROOT/'analysis/previous_campaign_stop_inventory.json',dict(time=time.time(),
        status='STOPPED_BY_RESEARCH_REDIRECTION',snapshot=freeze,copied_stop_records=copied,
        counts=dict(collections.Counter(r['status'] for r in inventory)),runs=inventory,
        unmaterialized_declared_conditions='NOT_RUN. Read the preserved frozen comparison and sweep declarations. No missing condition is imputed.',
        evidence='PROCESS_OWNERSHIP_AND_RAW_PROVENANCE_AUDIT'))
    distribution=importlib.metadata.distribution('vllm');rows=[];mismatches=[]
    for entry in distribution.files:
        if not str(entry).startswith('vllm/') or not str(entry).endswith('.py') or not entry.hash:continue
        path=distribution.locate_file(entry)
        actual=base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).decode().rstrip('=') if path.exists() else None
        passed=entry.hash.mode=='sha256' and actual==entry.hash.value
        row=dict(path=str(path),RECORD_hash=entry.hash.value,actual_RECORD_encoding=actual,passed=passed)
        rows.append(row)
        if not passed:mismatches.append(row)
    write(ROOT/'analysis/installed_vllm_wheel_source_integrity.json',dict(time=time.time(),
        version=distribution.version,checked_python_sources=len(rows),mismatches=mismatches,
        passed=bool(rows) and not mismatches,direct_url=json.loads(distribution.read_text('direct_url.json')),
        files=rows,evidence='INSTALLED_WHEEL_RECORD_VERIFICATION',
        scope='Verifies installed Python sources against this wheel RECORD. This does not audit every binary dependency or prove performance normality. No installed runtime file is changed.'))
    print('SAFE_STOP_INVENTORY',len(inventory),'VLLM_RECORD',len(rows),len(mismatches),flush=True)


if __name__=='__main__':main()
