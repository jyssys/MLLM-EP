"""Reconcile transport payload accounting with actual NIXL byte counters."""
import json
from .common import ROOT,write
from .gate import read
from .analyze import csv_write,quantile


def main():
    output=[]
    for folder in sorted((ROOT/'raw').glob('map_*')):
        if not (folder/'status.json').exists() or json.loads((folder/'status.json').read_text())['status']!='COMPLETE':continue
        cfg=json.loads((folder/'arc_contract.json').read_text())
        records=[r for p in folder.glob('migration_*.jsonl') for r in read(p)]
        for path in folder.glob('cohort_*_stats.json'):
            values=json.loads(path.read_text());contract=values['contract'];cohort=contract['cohort_id']
            window=json.loads((folder/f'window_{cohort}.json').read_text())
            payloads=[r for r in records if window['start']<=r['time']<=window['end']]
            actual=sum(values['after'][k]['transport_bytes_sent']-values['before'][k]['transport_bytes_sent'] for k in values['before'])
            accounted=sum(r['KV_bytes']+r['hidden_bytes'] for r in payloads)
            native=cfg.get('handoff_mode')=='native_G1'
            fragments=sum(values['after'][k]['transport_completed_count']-values['before'][k]['transport_completed_count'] for k in values['before'])
            hidden=(2048 if cfg['model']=='ouro' else 10560)*2*cfg['count'] if native else sum(r['hidden_bytes'] for r in payloads)
            output.append(dict(name=folder.name,model=cfg['model'],cohort=cohort,cut=contract['cut'],
                               handoff_mode=cfg.get('handoff_mode','cut_snapshot'),
                               actual_NIXL_bytes_sent=actual,KV_payload_bytes=actual-hidden if native else sum(r['KV_bytes'] for r in payloads),
                               hidden_payload_bytes=hidden,
                               payload_reconciled='NATIVE_TOTAL_COUNTER_ONLY' if native else actual==accounted,
                               source_exports=cfg['count'] if native else len(payloads),
                               fragments=fragments,descriptors='NOT_MEASURED' if native else sum(r['descriptors'] for r in payloads),
                               control_metadata_bytes='NOT_MEASURED',
                               evidence='MEASURED_NIXL_COUNTERS_AND_TRANSPORT_ACCOUNTING',
                               transfer_time='Host observation only. Pure DMA latency NOT_MEASURED.'))
    csv_write('migration_payloads.csv',output)
    matched=[r for r in output if r['handoff_mode']=='cut_snapshot']
    write(ROOT/'state/migration_validation.json',dict(rows=len(output),snapshot_reconciled=sum(r['payload_reconciled'] is True for r in matched),snapshot_rows=len(matched),native_counter_only_rows=len(output)-len(matched)))
    print('SNAPSHOT_PAYLOADS',sum(r['payload_reconciled'] is True for r in matched),'/',len(matched))


if __name__=='__main__':main()
