"""Separate checkpoint KV density from runtime allocation and measured latency."""
from .common import *
from .evidence import status


def density(model, data, depth):
    physical=data['n_layers'] if model=='huginn' else data['num_hidden_layers']
    if model=='huginn':
        valid=data['n_layers_in_prelude']+depth*data['n_layers_in_recurrent_block']+data['n_layers_in_coda']
    else:valid=physical*depth
    width=2*data['num_key_value_heads']*data.get('head_dim',data.get('hidden_size',0)//data.get('num_attention_heads',1))*2
    allocated=physical*depth
    return dict(physical_attention_layers=physical,fixed_recurrences=depth,
        valid_logical_attention_planes=valid,uniform_allocated_logical_planes=allocated,
        KV_heads=data['num_key_value_heads'],head_dim=data['head_dim'],BF16_bytes_per_scalar=2,
        KV_bytes_per_plane_token=width,fully_valid_KV_bytes_per_token=valid*width,
        uniform_reserved_KV_bytes_per_token=allocated*width,
        unused_uniform_plane_fraction=1-valid/allocated)


def analyze():
    from .analyze import save_csv
    values=[];allocations=[]
    for model,(repository,revision,depth) in MODEL_PINS.items():
        source=Path('/home/esjung/.cache/huggingface/hub')/('models--'+repository.replace('/','--'))/'snapshots'/revision/'config.json'
        data=json.loads(source.read_text());depth=1 if model=='qwen8' else depth
        row=dict(model=model,repository=repository,revision=revision,checkpoint_config_SHA256=digest(source),
            **density(model,data,depth),evidence='STRUCTURAL_ACCOUNTING_PINNED_CONFIG_AND_OBSERVED_LAYOUT',
            scope='Fully populated token accounting. Excludes block padding, metadata, temporary buffers, and valid-frontier exceptions. This is not a measured speedup or an isolated recurrence effect.')
        values.append(row)
        snapshot=ROOT/'configs'/f'pinned_architecture_{model}.json'
        if not snapshot.exists():write(snapshot,dict(repository=repository,revision=revision,source_SHA256=digest(source),checkpoint_config=data))
        for geometry in ROOT.glob('raw/capacity_'+model+'*/geometry.json'):
            if status(geometry.parent.name)!='COMPLETE':continue
            measured=json.loads(geometry.read_text())
            if not all(k in measured for k in ('KV_blocks','block_size')):continue
            actual=measured.get('KV_bytes',measured.get('KV_tensor_bytes'))
            if actual is None:continue
            expected=physical_pool_bytes=row['physical_attention_layers']*measured['KV_blocks']*measured['block_size']*row['KV_bytes_per_plane_token']
            allocations.append(dict(model=model,run=geometry.parent.name,KV_blocks=measured['KV_blocks'],block_size=measured['block_size'],
                measured_KV_allocation_bytes=actual,accounted_physical_pool_bytes=physical_pool_bytes,
                exact_allocation_accounting=actual==expected,evidence='MEASURED_ALLOCATION_PLUS_STRUCTURAL_ACCOUNTING',
                scope='Allocation identity only. Numerical parity and latency eligibility remain separate run gates.'))
    reference=next(v['fully_valid_KV_bytes_per_token'] for v in values if v['model']=='qwen8')
    for row in values:
        row.update(valid_KV_density_relative_to_Qwen=row['fully_valid_KV_bytes_per_token']/reference,
            reserved_KV_density_relative_to_Qwen=row['uniform_reserved_KV_bytes_per_token']/reference)
    save_csv('model_kv_density_accounting.csv',values);save_csv('kv_pool_allocation_identity.csv',allocations)
    write(ROOT/'model_kv_density_accounting.json',values);write(ROOT/'kv_pool_allocation_identity.json',allocations)
    return values


if __name__=='__main__':analyze()
