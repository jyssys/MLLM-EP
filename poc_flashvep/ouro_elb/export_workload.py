"""Freeze actual token IDs for mechanism probes. No held-out quality claim."""
import hashlib
from .common import ROOT,MODEL,write

def main():
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True)
    base=tokenizer.encode('A farmer measures a rectangle with length twelve and width seven. Explain the calculation carefully. '*1800)
    cases=[]
    for P in (128,256,2048,8192):
        for index in range(16):
            ids=base[index:index+P]
            cases.append(dict(id=f'rectangle_P{P}_offset{index}',source_family='rectangle_mechanism_CALIBRATION',
                split='CALIBRATION_NOT_HELD_OUT',P=P,offset=index,input_ids=ids))
    write(ROOT/'workload_manifest.json',dict(model_revision=MODEL.name,tokenizer_revision=MODEL.name,
        text_sha256=hashlib.sha256(('A farmer measures a rectangle with length twelve and width seven. Explain the calculation carefully. '*1800).encode()).hexdigest(),
        cases=cases,quality_claim='NONE',selector_train_val_final='NOT_RUN unless the P2 physical crossover gate passes',
        sampling=dict(temperature=0,exit_threshold=1.,max_loops=4,min_loops=4,precision='BF16'),
        screen_stop='64 generated tokens, EOS ignored for equal-length hardware characterization. Tokens are not forced.',
        quiet_stop='128 generated tokens, EOS ignored for equal-length hardware confirmation. Not complete-answer quality.',
        skipped_shape=dict(P=8192,N=16,reason='Four depth planes exceed the common 18,432-block KV pool. No eviction or capacity change.')))

if __name__=='__main__':main()
