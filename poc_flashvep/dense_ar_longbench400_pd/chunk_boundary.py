"""Clip a legal per-request prefill chunk before it crosses a2D boundary."""
import ast
import hashlib
import inspect
import linecache
import textwrap
from .common import write


def clip(computed,prefix,proposed):
    return min(proposed,prefix-computed) if computed<prefix else proposed


def install(adapter):
    if adapter.role!='decode' or not adapter.cfg.get('partition_aligned_chunks'):return
    from vllm.v1.core.sched.scheduler import Scheduler
    original=Scheduler.schedule;source=textwrap.dedent(inspect.getsource(original))
    tree=ast.parse(source);changed=[]
    class Transform(ast.NodeTransformer):
        def visit_Assign(self,node):
            if (len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='num_new_tokens'
                    and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='min'
                    and [ast.unparse(a) for a in node.value.args]==['num_new_tokens','token_budget']):
                extra=ast.parse('num_new_tokens = _campaign_clip_partition(request, num_new_tokens)').body[0]
                changed.append(node.lineno);return [node,ast.copy_location(extra,node)]
            return node
    tree=Transform().visit(tree);assert len(changed)==2,('Installed scheduler changed. Do not patch an unknown seam.',changed)
    def clipped(request,proposed):
        external=adapter.external(request.request_id)
        if external not in adapter.specs:return proposed
        part=adapter.partition(request.request_id)
        if not part.upper or not part.suffix:return proposed
        if adapter.cfg.get('fuse_short_hybrid') and part.length<=adapter.cfg['token_budget']:
            # The existing row-aware forward can compute suffix lower layers,
            # then upper layers for the complete legal query frame. Long
            # prompts retain the independently qualified aligned path.
            return proposed
        value=clip(request.num_computed_tokens,part.prefix,proposed)
        if value!=proposed:adapter.rows.append(dict(event='PARTITION_BOUNDARY_CHUNK',request_id=external,
            computed=request.num_computed_tokens,prefix=part.prefix,proposed=proposed,scheduled_upper=value))
        return value
    ast.fix_missing_locations(tree);namespace=dict(original.__globals__,_campaign_clip_partition=clipped)
    filename='<longbench400_partition_boundary_scheduler>'
    linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
    exec(compile(tree,filename,'exec'),namespace)
    Scheduler.schedule=namespace[original.__name__]
    write(adapter.folder/'partition_boundary_scheduler.json',dict(original_source_SHA256=hashlib.sha256(source.encode()).hexdigest(),
        changed_assignments=changed,evidence='SOURCE_AUDIT',
        scope='Only two per-request token counts are clipped before KV allocation. B8192 remains the upper bound. '
            'Native,Layer-only,Token-only and decode rows are unchanged. No post-allocation counter or assertion is bypassed.'))


if __name__=='__main__':
    assert clip(8192,12000,8192)==3808
    assert clip(12000,12000,8192)==8192
    assert clip(0,7000,8192)==7000
    print('PARTITION_BOUNDARY_CPU_PASS')
