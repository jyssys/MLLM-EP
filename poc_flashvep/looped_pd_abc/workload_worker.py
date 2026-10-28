"""Run the frozen worker with an explicit, archived heterogeneous input manifest."""
import hashlib
import json
import os
from pathlib import Path
from .common import ROOT,write

def main():
 from . import worker
 source=Path(os.environ['ABC_PROMPT_MANIFEST'])
 payload=source.read_bytes();rows=[json.loads(line) for line in payload.decode().splitlines() if line.strip()]
 model=os.environ['ABC_MANIFEST_MODEL'];name=os.environ['ABC_MANIFEST_RUN']
 assert rows and all(row['model']==model for row in rows)
 def manifest_prompts(requested_model,count,length):
  if requested_model!=model or length!=0 or count>len(rows):
   raise ValueError('The requested workload does not match its explicit manifest')
  return [list(row['input_ids']) for row in rows[:count]]
 worker.prompts=manifest_prompts
 write(ROOT/'raw'/name/'workload_provenance.json',dict(source=str(source),sha256=hashlib.sha256(payload).hexdigest(),
  rows=len(rows),model=model,scope='Synthetic padded questions with exact heterogeneous prompt lengths. Natural greedy outputs remain unfrozen.',
  engine_changes='No engine changes. Only the input provider differs from the frozen worker.',
  raw_prompt_source_override='synthetic_padded_GSM8K_train_validation_questions'))
 worker.main()

if __name__=='__main__':main()
