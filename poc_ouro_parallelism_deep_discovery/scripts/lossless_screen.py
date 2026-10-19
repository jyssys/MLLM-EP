"""CPU structural screen for exact recurrent-bit transport. Not codec latency."""
import argparse
import collections
import datetime
import math
import zlib
from poc_ouro_parallelism_deep_discovery.runtime import configure_imports
from poc_ouro_parallelism_deep_discovery.contract import *
configure_imports()
import torch

def entropy(data):
    counts=collections.Counter(data);N=len(data)
    return -sum((n/N)*math.log2(n/N) for n in counts.values()) if N else 0.

def bytes_of(value):
    return value.contiguous().view(torch.uint8).numpy().tobytes()

def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    assert not args.output.exists()
    tensors=torch.load(args.capture,map_location='cpu',weights_only=True)
    records=[]
    # Calls 1..4 are known full-R4 prefill executions. Decode boundaries depend
    # on online exit history, so only fixed decode pairs use the call-number rule.
    for name,value in tensors.items():
        prefix,call_text=name.rsplit('_call',1);call=int(call_text)
        if call<2:continue
        fixed=name.startswith('fixed_')
        if call<=4:phase='prefill'
        elif fixed and (call-5)%4!=0:phase='fixed_decode'
        else:continue
        prior=tensors.get(prefix+'_call'+str(call-1))
        if prior is None or prior.shape!=value.shape:continue
        assert value.dtype==prior.dtype==torch.bfloat16
        a,b=prior.view(torch.int16),value.view(torch.int16)
        xor=torch.bitwise_xor(a,b).contiguous()
        assert torch.equal(torch.bitwise_xor(a,xor),b)
        raw,delta=bytes_of(value),bytes_of(xor)
        # Byte-plane grouping is a reversible layout, not a floating-point delta.
        plane=xor.view(torch.uint8).reshape(-1,2).T.contiguous().numpy().tobytes()
        recon=torch.frombuffer(bytearray(plane),dtype=torch.uint8).reshape(2,-1).T.contiguous()
        assert torch.equal(recon.reshape_as(xor.view(torch.uint8)),xor.view(torch.uint8))
        records.append(dict(name=name,previous=prefix+'_call'+str(call-1),phase=phase,
            rows=value.shape[0],elements=value.numel(),raw_bytes=len(raw),
            bitwise_equal_fraction=float((a==b).float().mean()),
            byte_entropy_raw=entropy(raw),byte_entropy_XOR=entropy(delta),
            raw_zlib_bytes=len(zlib.compress(raw,level=1)),
            XOR_zlib_bytes=len(zlib.compress(delta,level=1)),
            XOR_byteplanes_zlib_bytes=len(zlib.compress(plane,level=1)),
            exact_reconstruction=True))
    assert records
    write(args.output,dict(records=records,capture=str(args.capture),capture_sha256=sha(args.capture),
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),evidence_class='ANALYTIC_BOUND',
        empirical_input='Measured pretrained activation samples, inspected on the CPU.',
        scope='Observed lossless encoded sizes, not GPU encoder/decoder or transport performance.',
        unknowns=['GPU processing latency','message startup','metadata','receiver base availability',
                  'end-to-end benefit','full-request representativeness'],
        sampling='Selected physical layers and first bounded calls. No benchmark-quality or KV compression claim.',
        precision='Original BF16 bits. XOR reconstruction is bitwise exact.'))
    print('LOSSLESS_SIZE_SCREEN',len(records),args.output)

if __name__=='__main__':main()
