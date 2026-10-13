"""CPU fixture exercises upstream states, not model TPF or performance."""
import sys
from types import SimpleNamespace
from poc_multibd_moe_ep.contract import *


def main():
    sys.path.insert(0,str(UPSTREAM))
    from diffulex.config import DecodingThresholds
    from diffulex.strategy_template.multi_block.engine.request import MultiBlockReqTemplate
    from diffulex.sampling_params import SamplingParams
    thresholds=DecodingThresholds(.9,.9,.95)
    rows=[]
    for r in [1,2,3,4]:
        config=SimpleNamespace(block_size=32,buffer_size=r,mask_token_id=156895,
            eos=156892,decoding_thresholds=thresholds,max_model_len=512)
        req=MultiBlockReqTemplate([100]*32,SamplingParams(temperature=0,max_tokens=256,max_nfe=128))
        req.init_multi_block(config)
        req.make_pending();req.step()
        # First prefill pass caches the prompt. A decode step activates slot0.
        req.postprocess();req.step()
        assert len(req.dllm_block_buffer.active_blocks)==1
        first=req.dllm_block_buffer.active_blocks[0]
        for position in range(29):
            first.write_token(101,position)
        req.step()
        active=len(req.dllm_block_buffer.active_blocks)
        assert active==(1 if r==1 else 2)
        assert req.chunk_size==32*r
        states=[dict(block=b.block_id,status=b.status.name,mask_tokens=b.num_mask_tokens)
                for b in req.dllm_block_buffer.dllm_blocks]
        first.write_token(101,29);first.write_token(101,30);first.write_token(101,31)
        first.commit_ready=True
        req.postprocess()
        assert first.is_to_cache
        req.postprocess()
        assert first.is_in_cache and first not in req.dllm_block_buffer.dllm_blocks
        rows.append(dict(R=r,physical_rows=32*r,active_after_29_of_32=active,
                         states_before_commit=states,cache_transition_pass=True))
    write(ROOT/'artifacts/cpu_state_machine.json',dict(pass_all=True,rows=rows,
        label='OFFICIAL_CPU_STATE_FIXTURE_NOT_MODEL_TPF',
        real_generation_state_machine_validation=False))
    print(rows)


if __name__=='__main__':main()
