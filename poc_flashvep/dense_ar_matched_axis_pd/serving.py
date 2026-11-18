"""Keep fixed cuts and use one token-budget admission predicate."""
import hashlib
import inspect
import json
import sys
from .common import ROOT, write, bind_legacy
from .configuration import destination_limit


def main():
    bind_legacy()
    cfg = json.load(open(sys.argv[1]))
    from ..dense_ar_slo_layer_pd import native
    if cfg['arm']=='DP2':
        return native.main()
    from ..dense_ar_slo_layer_pd import dynamic_native
    from ..dense_ar_layer_token_pd.recording import annotate
    dynamic_native.append = annotate(dynamic_native.append)
    code = inspect.getsource(dynamic_native.coordinator)
    needle = """if first_cut<36:
                        from .geometry import prefix_group
                        maximum=prefix_group([entry[1] for entry in list(pending_D)[:compatible]],input_lengths,
                            min(maximum,compatible),cfg.get('D_token_budget',cfg['token_budget']))"""
    replacement = "maximum=destination_limit([entry[1] for entry in list(pending_D)[:compatible]],cfg,min(maximum,compatible))"
    assert code.count(needle)==1
    transformed = code.replace(needle, replacement)
    if cfg.get('common_prospective_KV_reservation'):
        before="if choice.layers:\n                        generation=f'res_{group_number}'"
        after="if choice.layers or cfg.get('common_prospective_KV_reservation'):\n                        generation=f'res_{group_number}'"
        assert transformed.count(before)==1
        transformed=transformed.replace(before,after)
        # Never change the assigned static partition because memory is busy.
        # Retry the exact group after a rejection or a cancellation ACK.
        for reason in ('DESTINATION_CREDIT_REJECTED','NEGOTIATION_TIMEOUT_NATIVE_BEFORE_PREFIX'):
            before=f"job['choice'].layers=0;job['choice'].reason='{reason}'\n                            submit_prefix(job['indices'],job['choice'])"
            if reason.startswith('NEGOTIATION'):
                before=before.replace('\n                            submit_prefix','\n                        submit_prefix')
                indent='                        '
            else:
                indent='                            '
            after=("for retry_index in reversed(job['indices']):P_pending.appendleft(retry_index)\n"+
                indent+"P_since=time.monotonic()\n"+
                indent+"append(folder/'common_credit_retries.jsonl',dict(time=time.monotonic(),reason='"+reason+
                "',requests=[f'q{x}' for x in job['indices']],action='RETRY_SAME_FIXED_PARTITION_BEFORE_P'))")
            assert transformed.count(before)==1,reason
            transformed=transformed.replace(before,after)
    if cfg.get('source_chunk_packing'):
        before='if P_pending and not P_active and pending_negotiation is None:'
        after="if P_pending and not P_active and pending_negotiation is None and len(pending_D)+len(active_D)<cfg['source_outstanding_limit']:"
        assert transformed.count(before)==1
        transformed=transformed.replace(before,after)
        before="count=min(len(P_pending),cfg.get('P_group_size',cfg['P_slots']))"
        after="count=min(len(P_pending),cfg.get('P_group_size',cfg['P_slots']),cfg['source_outstanding_limit']-len(pending_D)-len(active_D))"
        assert transformed.count(before)==1
        transformed=transformed.replace(before,after)
    if not cfg.get('reuse_prefill_token'):
        # Reuse the predecessor's qualified placeholder-discard routing.
        # A token prefix or partial-layer prefix cannot supply a user token.
        for before,after,count in (
            ("assert cfg.get('reuse_prefill_token') and cfg['cap']>=2","assert cfg['cap']>=2",1),
            ('if cuts[rid]==36:',"if cuts[rid]==36 and cfg.get('reuse_prefill_token'):",2),
            ('offset=1 if cuts[rid]==36 else 0',
             "offset=1 if cuts[rid]==36 and cfg.get('reuse_prefill_token') else 0",1)):
            assert transformed.count(before)==count,before
            transformed=transformed.replace(before,after)
    scope = dict(dynamic_native.__dict__, destination_limit=destination_limit)
    exec(compile(transformed, __file__+':unified_admission', 'exec'), scope)
    dynamic_native.coordinator = scope['coordinator']
    write(ROOT/'raw'/cfg['name']/'unified_execution_contract.json', dict(
        original_SHA256=hashlib.sha256(code.encode()).hexdigest(),
        executed_SHA256=hashlib.sha256(transformed.encode()).hexdigest(),
        source=transformed, common_request_cap=cfg['D_admission_batch'],
        query_budget=8192, source_macro_limit=cfg['P_group_size'],
        source_chunk_packing=cfg.get('source_chunk_packing',False),
        source_outstanding_limit=cfg.get('source_outstanding_limit'),
        common_prospective_KV_reservation=cfg.get('common_prospective_KV_reservation',False),
        scope='Same FIFO, no-active-cohort and query-budget predicate in all PD arms. Actual cohort size can differ because query counts differ. No dynamic cut selection.'))
    dynamic_native.main()


if __name__ == '__main__':
    main()
