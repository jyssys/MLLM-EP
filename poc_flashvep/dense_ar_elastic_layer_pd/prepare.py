"""Freeze the diagnostic fixtures without treating viewed prompts as held-out."""
from .common import *


def from_previous(name,template,**changes):
    cfg=json.loads((PREVIOUS/'configs'/(template+'.json')).read_text());cfg.update(name=name,**changes)
    return cfg


def main():
    initialize()
    for name in ('qualification','calibration','discovery','heldout'):
        write(ROOT/'data'/(name+'.json'),json.loads((PREVIOUS/'data'/(name+'.json')).read_text()))
    write(ROOT/'data/prior_split_provenance.json',dict(base=BASE,
        prior_manifest=json.loads((PREVIOUS/'data/manifest.json').read_text()),
        scope='Previous heldout prompts were viewed. They are not fresh confirmation data for this campaign.'))
    files=[]
    for p in SOURCE.glob('*.py'):
        old='poc_flashvep/dense_ar_dynamic_layer_pd/'+p.name
        value=subprocess.run(['git','show',BASE+':'+old],cwd=REPO,capture_output=True)
        if value.returncode==0:files.append(dict(previous_path=old,previous_SHA256=hashlib.sha256(value.stdout).hexdigest(),
            new_path=str(p.relative_to(REPO)),initial_SHA256=digest(p)))
    write(ROOT/'state/executor_snapshot_origin.json',dict(base=BASE,files=files,
        scope='Separate namespace copy. Subsequent changes belong to this campaign. The previous source stays unchanged.'))
    cfg=from_previous('attention_fixed_tensor_batch_probe_v1','lean_native_O16_N64_v1',module='attention_probe',gpus=[0],
        count=1,request_caps=[16],arrival_offsets=[0.],phase='FIXED_QKV_ATTENTION_BATCH_DEPENDENCE_DIAGNOSTIC',wall_timeout=300)
    save(cfg)
    for name,template,changes in (
        ('repro_Native_O16_N8_v1','lean_native_O16_N64_v1',{}),
        ('repro_Continuous_chunk512_O16_N8_v1','HND_lean_chunk512_O16_N64_v1',{}),
        ('standard_Continuous_budget8192_O16_N32_v1','HND_lean_chunk512_O16_N64_v1',dict(chunk=8000,token_budget=8192))):
        count=32 if 'N32' in name else 8
        cfg=from_previous(name,template,count=count,dataset='discovery.json',
            request_caps=[16]*count,phase='PHASE1_CONTINUOUS_GPU_REPRODUCTION',**changes)
        cfg['arrival_offsets']=cfg['arrival_offsets'][:count];save(cfg)
    print('PREPARED_PHASE1')


if __name__=='__main__':main()
