"""CPU comparison of recorded owner tensor samples with exact HF shards."""
import json
import argparse
from safetensors import safe_open
from poc_multibd_moe_ep.contract import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--session',required=True)
    args=parser.parse_args()
    pins=json.loads((ROOT/'artifacts/pins.json').read_text())
    model=Path(pins['model_path'])
    index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
    checked=[]
    for artifact in sorted((ROOT/'artifacts').glob('R*/ownership.rank*.json')):
        if not artifact.parent.name.endswith('_'+args.session):continue
        owner=json.loads(artifact.read_text())
        for layer in owner['sparse_layers']+owner.get('tp_control_sparse_layers',[]):
            if 'weight_probe' not in layer:continue
            expert=owner['rank']*64
            for projection,actual in layer['weight_probe'].items():
                key=f"{layer['name']}.experts.{expert}.{projection}.weight"
                with safe_open(model/index[key],framework='pt',device='cpu') as inp:
                    expected=inp.get_slice(key)[0,:8].float().tolist()
                assert actual==expected,(artifact,key,actual,expected)
                checked.append(dict(artifact=str(artifact.relative_to(ROOT)),tensor=key,pass_exact=True))
    write(ROOT/'results'/args.session/'loaded_weight_identity.json',dict(status='PASS' if checked else 'NOT_RUN',
        checks=checked,scope='8 exact BF16 values/projection for first local expert at sparse layers 1 and 19',
        no_full_tensor_hash_claim=True,checkpoint=MODEL_ID,revision=MODEL_REVISION))
    print('Exact loaded-owner sample checks:',len(checked))


if __name__=='__main__':main()
