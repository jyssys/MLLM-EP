"""Prepare the exact text decoder before the first GPU response."""
import hashlib,json,time


def primed_stream(tokenizer,ids,skip_special_tokens=True):
    from tokenizers.decoders import DecodeStream
    assert ids
    stream=DecodeStream(ids=ids[:-1],skip_special_tokens=skip_special_tokens)
    stream.step(tokenizer,ids[-1])
    return stream


def install(event,verify=False):
    from vllm.v1.engine.detokenizer import FastIncrementalDetokenizer
    original_init=FastIncrementalDetokenizer.__init__;original_decode=FastIncrementalDetokenizer.decode_next
    def initialized(self,tokenizer,request):
        original_init(self,tokenizer,request)
        start=time.monotonic();ids=request.prompt_token_ids
        if not ids:return
        self.stream=primed_stream(self.tokenizer,ids,self.skip_special_tokens)
        self._dense_shadow=None;self._dense_external=request.external_req_id
        if verify:
            shadow=FastIncrementalDetokenizer.__new__(FastIncrementalDetokenizer)
            original_init(shadow,tokenizer,request);self._dense_shadow=shadow
        event('EXACT_TEXT_DECODER_PRIMED',request_id=request.request_id,external_request_id=request.external_req_id,
            prompt_tokens=len(ids),host_elapsed_ms=1000*(time.monotonic()-start),
            scope='The final prompt token primes the original decode stream. No generated token is consumed. Necessary text preparation moves before GPU execution.')
    def decoded(self,next_token_id):
        text=original_decode(self,next_token_id);shadow=getattr(self,'_dense_shadow',None)
        if shadow is not None:
            expected=original_decode(shadow,next_token_id)
            assert text==expected,'Primed and original streamed text differ'
            event('PRIMED_TEXT_PARITY',external_request_id=self._dense_external,token_id=next_token_id,
                passed=True,text_SHA256=hashlib.sha256(text.encode()).hexdigest())
        return text
    FastIncrementalDetokenizer.__init__=initialized
    FastIncrementalDetokenizer.decode_next=decoded


def cpu_check():
    from tokenizers import Tokenizer
    from tokenizers.decoders import DecodeStream
    from .common import ROOT,MODEL,write
    from .analyze import records
    tokenizer=Tokenizer.from_file(str(MODEL/'tokenizer.json'));rows=records(ROOT/'raw/eager_confirm_qwen8_L8000_O16_N128_s11_native_v25/requests.jsonl')
    checked=[];start=time.monotonic()
    for r in rows:
        for native in (False,True):
            ids=r['input_ids']+r['output_ids'][:1] if native else r['input_ids']
            outputs=r['output_ids'][1:] if native else r['output_ids']
            original=DecodeStream(ids=ids,skip_special_tokens=True);primed=primed_stream(tokenizer,ids)
            a=[original.step(tokenizer,t) for t in outputs];b=[primed.step(tokenizer,t) for t in outputs]
            assert a==b,(r['sample_id'],native)
            checked.append(dict(sample_id=r['sample_id'],native_token_bridge=native,output_tokens=len(outputs),passed=True))
    result=dict(passed=len(checked)==256,cases=checked,elapsed_seconds=time.monotonic()-start,
        evidence='CPU_TEXT_EXACTNESS_ONLY',scope='All 128 real prompts, both original and P-token-extended D prompts. Every incremental output string matches the unmodified decoder. No GPU performance result.')
    write(ROOT/'analysis/decoder_priming_CPU_gate.json',result);print('PRIMING_CPU_PASS',len(checked))


if __name__=='__main__':cpu_check()
