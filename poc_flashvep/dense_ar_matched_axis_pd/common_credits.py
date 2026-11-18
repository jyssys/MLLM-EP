"""Reuse one prospective KV contract for Native, Layer and Token PD."""
import hashlib
import inspect
import textwrap
from .common import ROOT,write


def transform(source):
    old='cut in (24,28,32,34)'
    assert source.count(old)==1
    return source.replace(old,'cut in (24,28,32,34,36)')


def install(cfg,role):
    if not cfg.get('common_prospective_KV_reservation'):
        return
    from ..dense_ar_slo_layer_pd import credits
    if not getattr(credits.Credits.reserve,'_matched_generalized',False):
        original=textwrap.dedent(inspect.getsource(credits.Credits.reserve))
        source=transform(original)
        scope=dict(credits.__dict__)
        exec(compile(source,__file__+':existing_credits_with_native_cut','exec'),scope)
        method=scope['reserve'];method._matched_generalized=True
        method._matched_original_source=original
        method._matched_source=source
        credits.Credits.reserve=method
    method=credits.Credits.reserve
    write(ROOT/'raw'/cfg['name']/f'common_KV_credit_contract_{role}.json',dict(
        common_prospective_reservation=True,full_context_blocks=True,
        original_SHA256=hashlib.sha256(method._matched_original_source.encode()).hexdigest(),
        executed_SHA256=hashlib.sha256(method._matched_source.encode()).hexdigest(),
        source=method._matched_source,
        scope='Keep the existing reservation, commit, allocator exclusion, request mapping and retirement code. Only permit cut36 in the existing cut assertion. No new KV manager. All PD methods reserve the full destination prompt/output block demand before P work.'))
