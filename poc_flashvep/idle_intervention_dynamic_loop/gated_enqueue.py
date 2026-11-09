"""Admit a larger Token cohort only after the preceding exact-state gate."""
from .common import *
from .qualification import qualify


def main():
    assert RESUMPTION is not None
    gates={
        'r9_ouro_actual_mixed_state32':'r9_ouro_actual_mixed_state8',
        'r9_huginn_actual_mixed_state32':'r9_huginn_actual_mixed_state8',
        'r7_huginn_actual_mixed_state32':'r7_huginn_actual_mixed_state8',
        'r6_ouro_actual_mixed_state32':'r6_ouro_actual_mixed_state8',
        'r6_huginn_actual_mixed_state32':'r6_huginn_actual_mixed_state8',
        'r6_ouro_loop_selection32':'r9_ouro_actual_mixed_state32',
        'r6_ouro_tile_selection32':'r9_ouro_actual_mixed_state32',
        'r6_ouro_token_selection32':'r4_ouro_actual_token_state32',
        'r6_huginn_loop_selection32':'r9_huginn_actual_mixed_state32',
        'r6_huginn_tile_selection32':'r9_huginn_actual_mixed_state32',
        'r6_huginn_token_selection32':'r4_huginn_actual_token_state32',
        'r5_ouro_actual_mixed_state32':'r5_ouro_actual_mixed_state8',
        'r4_ouro_actual_token_state32':'r4_ouro_actual_token_state8',
        'r3_huginn_actual_token_state32':'r3_huginn_actual_token_state8',
        'r1_ouro_token_selection32':'r4_ouro_actual_token_state32',
        'r1_huginn_token_selection32':'r4_huginn_actual_token_state32',
    }
    path=ROOT/'state/queue.json';queue=json.loads(path.read_text());ready=[]
    for name,dependency in gates.items():
        if name in queue:continue
        cfg=json.loads((ROOT/'configs'/(dependency+'.json')).read_text());model=cfg['model']
        reference=f'a_{model}_native_state8' if cfg['count']==8 else f'n32_{model}_native_full_state32'
        q=qualify(dependency,reference)
        if q.get('passed') and (not name.startswith('r9_') or q.get('actual_dynamic_continuation_tested')):ready.append(name)
    active=json.loads((ROOT/'state/active.json').read_text()).get('name')
    index=queue.index(active)+1 if active in queue else len(queue)
    write(path,queue[:index]+ready+queue[index:])
    print(json.dumps(dict(admitted=ready,dependencies=gates)))


if __name__=='__main__':main()
