"""Two surgical opt-in repairs; never mutate the pinned external checkout.

The upstream EP guard is bypassed only for diagnostic bring-up. A separate
owner-local-ID repair tests the installed normal DeepEP API, not decoding logic.
These changes do not themselves establish full-model EP validity.
"""
import ast
import inspect
import textwrap


def enable_config_ep():
    import diffulex.config as module
    source = textwrap.dedent(inspect.getsource(module.Config.__post_init__))
    tree = ast.parse(source)
    function = tree.body[0]
    guards = [n for n in function.body if isinstance(n,ast.If) and
              'MoE A2A backends are currently unsupported' in ast.unparse(n)]
    assert len(guards) == 1, 'Pinned unsupported-path guard changed'
    function.body.remove(guards[0])
    scope = dict(vars(module))
    exec(compile(ast.fix_missing_locations(tree), '<poc-ep-guard-only>', 'exec'), scope)
    module.Config.__post_init__ = scope['__post_init__']


def repair_normal_owner_local_ids():
    import diffulex.moe.dispatcher.deepep_dispatcher as module
    source = textwrap.dedent(inspect.getsource(module.DeepEPDispatcher.dispatch_b))
    original = '''local_mask = (native_recv_topk_ids >= self.local_expert_start) & (
            native_recv_topk_ids < self.local_expert_end
        )'''
    fixed = '''local_mask = (native_recv_topk_ids >= 0) & (
            native_recv_topk_ids < self.num_local_experts
        )'''
    assert source.count(original) == 1
    source = source.replace(original, fixed)
    original_sub = 'native_recv_topk_ids[local_mask] - self.local_expert_start'
    assert source.count(original_sub) == 1
    source = source.replace(original_sub, 'native_recv_topk_ids[local_mask]')
    scope = dict(vars(module))
    exec(compile(source, '<poc-owner-local-id-repair>', 'exec'), scope)
    module.DeepEPDispatcher.dispatch_b = scope['dispatch_b']
