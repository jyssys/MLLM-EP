"""Keep the exact layer runtime and make geometry controls explicit."""
import json
import os
import sys
from .common import ROOT, SLO_PRIOR, digest, write


def main():
    from ..dense_ar_slo_layer_pd import dynamic_native
    from .recording import annotate
    dynamic_native.append = annotate(dynamic_native.append)
    cfg = json.load(open(sys.argv[1]))
    from .flow_policy import install
    install(cfg, dynamic_native)
    from .geometry_control import group_size
    dynamic_native.declared_source_group_size=group_size
    if cfg.get('preplanned_source_groups') or not cfg.get('reuse_prefill_token') or cfg.get('flow_balance_policy'):
        import hashlib
        import inspect
        code = inspect.getsource(dynamic_native.coordinator)
        replacements = []
        if cfg.get('preplanned_source_groups'):
            replacements.append(('if P_pending and not P_active and pending_negotiation is None:',
                'if P_pending and not P_active and pending_negotiation is None and '
                "(len(P_pending)>=declared_source_group_size(P_pending[0],cfg) or i==len(data)):", 1))
            replacements.append(("count=min(len(P_pending),cfg.get('P_group_size',cfg['P_slots']))",
                'count=min(len(P_pending),declared_source_group_size(P_pending[0],cfg))',1))
        if cfg.get('flow_balance_policy'):
            replacements.append(('decision_start=time.monotonic();choice=policy.choose(decision_state)',
                "visible=list(P_pending);size=cfg['P_group_size']\n"
                "                    decision_state.visible_fifo_groups=tuple(dict(indices=visible[j:j+size],caps=tuple(cfg['request_caps'][x] for x in visible[j:j+size]),arrivals=tuple(arrivals[f'q{x}'] for x in visible[j:j+size])) for j in range(0,min(len(visible)//size,cfg.get('flow_horizon_groups',3)-1)*size,size))\n"
                '                    decision_start=time.monotonic();choice=policy.choose(decision_state)', 1))
        if not cfg.get('reuse_prefill_token'):
            replacements.extend([
                ("assert cfg.get('reuse_prefill_token') and cfg['cap']>=2", "assert cfg['cap']>=2", 1),
                ('if cuts[rid]==36:', "if cuts[rid]==36 and cfg.get('reuse_prefill_token'):", 2),
                ('offset=1 if cuts[rid]==36 else 0',
                 "offset=1 if cuts[rid]==36 and cfg.get('reuse_prefill_token') else 0", 1)])
        transformed = code
        for needle, replacement, expected in replacements:
            assert transformed.count(needle) == expected, needle
            transformed = transformed.replace(needle, replacement)
        namespace = dict(dynamic_native.__dict__)
        exec(compile(transformed, __file__ + ':preplanned_source_groups', 'exec'), namespace)
        dynamic_native.coordinator = namespace['coordinator']
        write(ROOT / 'raw' / cfg['name'] / 'geometry_execution_contract.json', dict(
            original_coordinator_SHA256=hashlib.sha256(code.encode()).hexdigest(),
            executed_coordinator_SHA256=hashlib.sha256(transformed.encode()).hexdigest(),
            group_size=cfg['P_group_size'], final_partial_group=True,
            transformations=[dict(before=a, after=b, occurrences=n) for a, b, n in replacements],
            executed_coordinator_source=transformed,
            scope='Explicit adapter. Native source-token reuse is disabled only when it cannot produce the final prompt output, or in a declared correctness control. Optional matched geometry waits for declared FIFO members in every arm. This is not a globally strongest admission claim.'))
    dynamic_native.main()


if __name__ == '__main__':
    main()
