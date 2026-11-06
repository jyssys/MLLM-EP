"""Audit actual work, local ownership, credits, and storage without time sums."""
from .common import *
from .analyze import table,requests


def main():
    work=[];memory=[];contracts=[]
    for path in sorted((ROOT/'configs').glob('*.json')):
        cfg=json.loads(path.read_text())
        if 'name' not in cfg:continue
        folder=ROOT/'raw'/cfg['name'];total=None
        for p in folder.glob('native_contract_final_*.json'):
            value=json.loads(p.read_text());contracts.append(dict(name=cfg['name'],model=cfg['model'],**value))
        for p in folder.glob('memory_inventory_close_*.json'):
            value=json.loads(p.read_text())
            # Use the physical capacity reported by this run's NVML inventory.
            total=None
            isolation=folder/'isolation.json'
            if isolation.exists():
                text=json.loads(isolation.read_text()).get('GPU_query','')
                for line in text.splitlines()[1:]:
                    parts=[s.strip() for s in line.split(',')]
                    if len(parts)>3 and parts[0].isdigit():
                        total=int(parts[3].split()[0])*2**20;break
            for field,amount in value.items():
                if not field.endswith('_bytes'):continue
                memory.append(dict(name=cfg['name'],model=cfg['model'],role=value['role'],component=field,
                    bytes=amount,GiB=amount/2**30,physical_HBM_bytes=total,
                    HBM_percent=100*amount/total if total else None,
                    scope='Unique-storage components overlap allocator totals. Do not sum both. Capacity uses NVML physical HBM, not the rounded product label.'))
        before=folder/'before_measured.json';after=folder/'after_measured.json'
        if not before.exists() or not after.exists():continue
        a,b=json.loads(before.read_text()),json.loads(after.read_text())
        rows=requests(folder)
        locals=[];refusals={};released=[];max_local_credit=0;local_reservations={}
        for event_file in folder.glob('dlocal_events_*.json'):
            for e in json.loads(event_file.read_text())['events']:
                rid=e.get('request_id','')
                if e['event']=='PLACEMENT' and rid.startswith('r0_q') and e['local']:locals.append(rid)
                if e['event']=='LOCAL_RETIRED':released.append(e)
                if e['event']=='LOCAL_ADMISSION_DEFERRED':
                    refusals[e['reason']]=refusals.get(e['reason'],0)+1
                if e['event']=='LOCAL_CREDIT_RESERVED':
                    max_local_credit=max(max_local_credit,e['blocks'])
                    if rid.startswith('r0_q'):local_reservations[e['tid']]=rid
        retired={e['tid'] for e in released}
        completed_local={rid for tid,rid in local_reservations.items() if tid in retired}
        result=dict(name=cfg['name'],model=cfg['model'],method=cfg.get('dlocal_method'),completed=len(rows),
            offered=cfg['count'],selected_local_requests=len(set(locals)),admitted_local_requests=len(local_reservations),
            actual_local_serving_requests=len(completed_local),
            max_single_local_reserved_blocks=max_local_credit,
            local_admission_observations=refusals,
            observation_scope='Deferred-credit log samples are throttled at 250 ms per request. They are not a total refusal-attempt count.',
            expected_total_recurrent_rows=sum((r['input_tokens']+r['output_tokens']-1)*cfg['depth'] for r in rows))
        for snapshot in folder.glob('memory_progress_*.jsonl'):
            with snapshot.open() as stream:
                line=stream.readline()
            if not line:continue
            value=json.loads(line)
            if value['role']!='decode':continue
            quota=max_local_credit*value['KV_block_bytes']
            result.update(D_KV_block_bytes=value['KV_block_bytes'],D_KV_total_blocks=value['KV_total_blocks'],
                single_local_reservation_GiB=quota/2**30,
                single_local_reservation_physical_HBM_percent=100*quota/total if total else None,
                pool_only_theoretical_local_reservations=value['KV_total_blocks']//max_local_credit if max_local_credit else None,
                reservation_scope='Measured reservation quota times native bytes per block. Pool-only quotient ignores existing native reservations and compute slots.')
            break
        total_work=0;valid_end=True
        for key,initial in a.items():
            if key not in b:valid_end=False;continue
            final=b[key];role='P' if 'prefill' in key else 'D'
            amount=final['issued_loop_rows']-initial['issued_loop_rows'];total_work+=amount
            result[role+'_recurrent_rows']=amount
            result[role+'_submissions']=final['issued_submissions']-initial['issued_submissions']
            result[role+'_NIXL_sent_bytes']=final['transport_bytes_sent']-initial['transport_bytes_sent']
            valid_end=valid_end and final['requests']==0 and final['used_blocks']==0
        result.update(actual_total_recurrent_rows=total_work,
            full_fixed_work_exact=total_work==result['expected_total_recurrent_rows'],
            final_device_requests_and_KV_blocks_empty=valid_end,
            local_prompt_recurrent_rows=len(completed_local)*cfg['length']*cfg['depth'],
            row_scope='Real recurrent token rows. Not FLOPs. Prelude/Coda and padding work are not included.')
        work.append(result)
    table(ROOT/'work_accounting.csv',work);table(ROOT/'gpu_memory_inventory.csv',memory)
    write(ROOT/'native_contracts.json',contracts)
    print(json.dumps(dict(work_runs=len(work),memory_rows=len(memory),native_contracts=len(contracts))))


if __name__=='__main__':main()
