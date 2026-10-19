"""Read-only hardware, dependency, and checkpoint inventory."""
import datetime
import importlib.metadata
import struct
import subprocess
from poc_ouro_parallelism_deep_discovery.contract import *

def command(args):return subprocess.check_output(args,text=True).strip()

def header(path):
    with path.open('rb') as stream:
        N=struct.unpack('<Q',stream.read(8))[0]
        values=json.loads(stream.read(N))
    tensors={k:v for k,v in values.items() if k!='__metadata__'}
    return dict(file_bytes=path.stat().st_size,header_bytes=N,
        tensor_count=len(tensors),payload_bytes=sum(v['data_offsets'][1]-v['data_offsets'][0] for v in tensors.values()),
        dtype_counts=dict(__import__('collections').Counter(v['dtype'] for v in tensors.values())),
        selected={k:v for k,v in tensors.items() if k.startswith('model.layers.0.') or 'early_exit_gate' in k})

def main():
    output=ROOT/'artifacts/environment_manifest.json'
    assert not output.exists()
    native=Path(NATIVE_PYTHON).parents[1]/'lib/python3.14/site-packages'
    versions={}
    for name in ['torch','vllm','transformers','flashinfer-python']:
        found=[d for d in importlib.metadata.distributions(path=[str(native)])
               if d.metadata['Name'].lower().replace('_','-')==name]
        versions[name]=[d.version for d in found]
    models={size:dict(id=info['id'],revision=info['revision'],
        config=read(info['path']/'config.json'),config_sha256=sha(info['path']/'config.json'),
        tokenizer_config_sha256=sha(info['path']/'tokenizer_config.json'),
        modeling_sha256=sha(info['path']/'modeling_ouro.py'),
        weights_header=header(info['path']/'model.safetensors')) for size,info in MODELS.items()}
    write(output,dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        hardware=command(['nvidia-smi','--id=4,5,6,7',
            '--query-gpu=index,uuid,name,memory.total,driver_version,pci.bus_id,power.limit',
            '--format=csv,noheader']),topology=command(['nvidia-smi','topo','-m']),
        process_inventory_scope='Per-run ownership guards retain private process inventories. No foreign process is signalled.',
        physical_pool=GPUS,UUIDS=UUIDS,P2_mapping=[4,6],P4_mapping=GPUS,
        topology_scope='Native dense TP. EP1/PP1/DP1/CP1 for native TP baselines. CDB quality uses one independent GPU.',
        versions=versions,models=models,CDB_revision=CDB_REVISION,
        CDB_source_revision_observed=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CDB,text=True).strip(),
        environment_scope='Separate interpreter. Existing binary dependencies are read-only. No package upgrades.',
        model_weights_runtime='Pretrained BF16, no precision reduction or new trained component.',
        observation_scope='Inventory, not HBM traffic or performance counters.'))
    print('ENVIRONMENT_SAVED',output)

if __name__=='__main__':main()
