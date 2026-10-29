"""Check the timing protocol separately from model correctness and source identity."""
import json
from .common import ROOT

def check(folder):
 epoch=ROOT/'configs/evidence_protocol_epoch.json';path=folder/'evidence_protocol.json'
 if not epoch.exists():return dict(valid=False,mode='legacy_sync',reason='Nonblocking evidence epoch is not registered')
 if not path.exists():return dict(valid=False,mode='legacy_sync',reason='Foreground synchronous evidence I/O. Retiming required.')
 reference=json.loads(epoch.read_text());record=json.loads(path.read_text())
 if record.get('mode')!='async_nonblocking' or record.get('telemetry_sha256')!=reference['telemetry_sha256']:
  return dict(valid=False,mode=record.get('mode'),reason='Evidence implementation differs from the registered epoch')
 config=json.loads((folder/'configuration.json').read_text());attempt=config.get('arguments',{}).get('attempt_id')
 proofs=[]
 for proof in folder.glob('async_evidence_writer_*.json'):
  data=json.loads(proof.read_text())
  if data.get('attempt_id')!=attempt:continue
  if data.get('telemetry_sha256')!=reference['telemetry_sha256']:
   return dict(valid=False,mode=record['mode'],reason='A writer uses a different recorder implementation')
  summary=folder/f"async_evidence_summary_{data['pid']}.json"
  if not summary.exists():return dict(valid=False,mode=record['mode'],reason='An evidence writer has no final durable summary')
  stats=json.loads(summary.read_text())
  if stats.get('attempt_id')!=attempt or stats.get('telemetry_sha256')!=reference['telemetry_sha256']:
   return dict(valid=False,mode=record['mode'],reason='A writer summary belongs to a different attempt or recorder')
  if stats.get('error') or stats.get('enqueued_records')!=stats.get('background_written_records'):
   return dict(valid=False,mode=record['mode'],reason='Evidence records are incomplete')
  proofs.append(data)
 if not proofs:return dict(valid=False,mode=record['mode'],reason='No durable writer proof')
 for failure in folder.glob('async_evidence_failure_*.json'):
  if json.loads(failure.read_text()).get('attempt_id')==attempt:
   return dict(valid=False,mode=record['mode'],reason='An evidence writer failed')
 return dict(valid=True,mode=record['mode'],reason='Registered nonblocking protocol. Model fidelity remains a separate gate.')
