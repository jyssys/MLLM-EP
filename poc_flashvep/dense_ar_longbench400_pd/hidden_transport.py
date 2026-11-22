"""Check exact raw hidden/residual payloads in correctness captures only."""
import hashlib
import json
from ..dense_ar_bi0_shortanswer_pd.variable_hidden import HiddenTransport as Original
from .common import write


class HiddenTransport(Original):
    def __init__(self,cfg,role,folder):
        reserve=cfg.get('hidden_pool_free_reserve_bytes')
        if reserve is not None:
            import torch
            free_before_reclaim,_=torch.cuda.mem_get_info()
            cached_unused=max(0,torch.cuda.memory_reserved()-torch.cuda.memory_allocated())
            if cfg.get('hidden_pool_reclaim_unused_cache'):
                # Initialization only. Never free a live tensor or change KV42GiB.
                # Raw cuda free memory alone does not include reclaimable cache.
                torch.cuda.empty_cache()
            slots=cfg['hidden_source_slots' if role=='prefill' else 'hidden_destination_slots']
            required=slots*2*38912*4096*2
            free,total=torch.cuda.mem_get_info()
            passed=free>=required+reserve
            write(folder/f'hidden_pool_memory_{role}.json',dict(slots=slots,
                raw_free_before_optional_reclaim_bytes=free_before_reclaim,
                cached_unused_before_optional_reclaim_bytes=cached_unused,
                unused_cache_reclaimed=bool(cfg.get('hidden_pool_reclaim_unused_cache')),
                free_before_pool_bytes=free,total_device_bytes=total,pool_bytes=required,
                required_free_after_pool_bytes=reserve,passed=passed,evidence='MEMORY_ADMISSION_CHECK',
                scope='Actual free device memory before hidden-pool allocation. '
                    'Reject a candidate before allocation when its frozen workspace reserve is unavailable. '
                    'This is not a measured peak-memory or performance result. KV42GiB remains unchanged.'))
            assert passed,'HIDDEN_POOL_MEMORY_RESERVE_NOT_AVAILABLE'
        super().__init__(cfg,role,folder)

    def checksum(self,slot,length):
        import torch
        if self.cfg.get('gpu_state_digest'):
            from .gpu_digest import gpu
            return [gpu(self.pool[slot,i,:length],verify=self.cfg.get('verify_GPU_digest_with_CPU',False))['digest'] for i in (0,1)]
        return [hashlib.sha256(self.pool[slot,i,:length].detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest() for i in (0,1)]

    def put_part(self,rid,start,length,hidden,residual,cut):
        super().put_part(rid,start,length,hidden,residual,cut)
        if self.cfg.get('axis_state_capture') and self.parts[rid]['cursor']==length:
            slot=self.parts[rid]['slot']
            from .gpu_digest import ALGORITHM
            algorithm=ALGORITHM if self.cfg.get('gpu_state_digest') else 'RAW_SHA256'
            field='hidden_residual_digests' if self.cfg.get('gpu_state_digest') else 'hidden_residual_SHA256'
            write(self.directory/(rid+'.source_tensor_hashes.json'),dict(request_id=rid,shape=[length,4096],digest_algorithm=algorithm,
                dtype='BF16',**{field:self.checksum(slot,length)},bytes=length*16384,
                evidence='CORRECTNESS_ONLY',scope='Full valid raw hidden and residual tensors. CPU copies invalidate timing.'))

    def get(self,rid,cut):
        first=rid not in self.received;value=super().get(rid,cut)
        if first and self.cfg.get('axis_state_capture'):
            if not self.cfg.get('axis_import_hash_after_model'):self.verify_parity(rid)
        return value

    def verify_parity(self,rid):
        from .gpu_digest import ALGORITHM
        slot,length=self.received[rid];source=json.loads((self.directory/(rid+'.source_tensor_hashes.json')).read_text())
        algorithm=ALGORITHM if self.cfg.get('gpu_state_digest') else 'RAW_SHA256'
        assert source.get('digest_algorithm','RAW_SHA256')==algorithm
        expected=source['hidden_residual_digests' if self.cfg.get('gpu_state_digest') else 'hidden_residual_SHA256']
        actual=self.checksum(slot,length);passed=actual==expected
        write(self.directory/(rid+'.destination_tensor_parity.json'),dict(request_id=rid,passed=passed,bytes=length*16384,
            digest_algorithm=algorithm,source=expected,destination=actual,
            check_after_prefix_consumption=bool(self.cfg.get('axis_import_hash_after_model')),evidence='CORRECTNESS_ONLY'))
        assert passed,'Different raw hidden or residual tensors. Preserve the failed run.'

    def retire(self,rid):
        if self.cfg.get('axis_state_capture') and self.cfg.get('axis_import_hash_after_model'):self.verify_parity(rid)
        return super().retire(rid)
