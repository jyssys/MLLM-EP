"""Preserve requested/effective Huginn mode differences without bypassing guards."""
from .common import *


def apply():
    marker=ROOT/'state/huginn_requested_async_not_effective_20261102.json'
    if marker.exists():return
    from .campaign import terminal
    worker=REPO/'poc_flashvep/adaptive_loop_token_pd_2gpu/worker.py'
    engine=RLT/'vllm_rlt/engine/stage_engine.py'
    assert "if cfg['model']=='ouro' and cfg.get('asynchronous',True)" in worker.read_text()
    assert 'Huginn requires synchronous scheduling without prefill UVA' in engine.read_text()
    path=ROOT/'state/skipped_jobs.json';skips=json.loads(path.read_text()) if path.exists() else {};retired=[]
    for p in (ROOT/'configs').glob('*.json'):
        cfg=json.loads(p.read_text())
        if not isinstance(cfg,dict) or cfg.get('model')!='huginn' or not cfg.get('asynchronous') or not cfg.get('name'):continue
        if terminal(cfg['name'])=='PENDING':
            skips[cfg['name']]=dict(time=time.time(),status='NOT_RUN_UNSUPPORTED_OR_IGNORED_HUGINN_ASYNC',
                reason='Direct StageEngine rejects async. The inherited native-PD launcher applies the async CLI flag only to Ouro. Do not label a Sync run as an Async experiment.')
            retired.append(cfg['name'])
    write(path,skips)
    p=ROOT/'state/extension_runtime_controls_20261102.json';old=json.loads(p.read_text()) if p.exists() else {}
    saved={}
    for key,value in old.items():
        if value.get('model')=='huginn' and value.get('source_config',{}).get('asynchronous'):
            saved[key]=value;old[key]=dict(value,stage='UNSUPPORTED_OR_IGNORED_ASYNC_REQUEST')
    write(p,old)
    prior={}
    for name in ('extension_transport_20261102.json','extension_huginn_slot_controls_20261102.json'):
        p=ROOT/'state'/name
        if not p.exists():continue
        value=json.loads(p.read_text());prior[name]=value
        if name.startswith('extension_transport'):
            value.pop('huginn',None);write(p,value)
        else:write(p,dict(stage='INITIAL'))
    write(marker,dict(time=time.time(),status='SOURCE_VERIFIED_MODE_MISMATCH',retired_pending_jobs=retired,
        old_runtime_states=saved,old_transport_and_slot_states=prior,
        source_evidence=[dict(path=str(worker),SHA256=digest(worker),finding='Only Ouro receives --asynchronous in the native-PD launcher.'),
            dict(path=str(engine),SHA256=digest(engine),finding='The actual Huginn StageEngine rejects async_scheduling=True.')],
        GPU_feedback='capacity_huginn_ext_async_l3900_g4_full_p_v1 failed with the native synchronous-only guard.',
        scope='Completed native PD outputs, state bytes, and timings remain real Sync measurements. They are not evidence of supported Huginn Async. No runtime guard is removed. No active process is stopped.'))


if __name__=='__main__':apply()
