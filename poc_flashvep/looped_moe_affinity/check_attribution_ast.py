"""Check observer expression preservation. This is not a GPU correctness test."""
import ast
import builtins
import copy
import inspect
import textwrap
from types import SimpleNamespace
from unittest.mock import patch
from .common import ROOT,write
from .attribution_markers import Markers,instrument_ep
from ..looped_moe_ep import runtime

class RemoveObserver(ast.NodeTransformer):
    def __init__(self,remove_alias=True):self.remove_alias=remove_alias

    def visit_With(self,node):
        node=self.generic_visit(node)
        if len(node.items)==1 and 'self.attribution.region' in ast.unparse(node.items[0].context_expr):
            return node.body
        return node

    def visit_Assign(self,node):
        if self.remove_alias and runtime.assign_name(node)=='source_rows':return None
        return self.generic_visit(node)

    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='self.attribution.record_counts':return None
        return self.generic_visit(node)

    def visit_Subscript(self,node):
        node=self.generic_visit(node)
        if self.remove_alias and isinstance(node.slice,ast.Name) and node.slice.id=='source_rows':
            node.slice=ast.parse('x[self.rank::self.world]').body[0].value.slice
        return node

def main():
    original=ast.parse(textwrap.dedent(inspect.getsource(runtime.PhysicalEP._run)))
    original_dump=ast.dump(original,include_attributes=False)
    checked=[]
    real_compile=builtins.compile
    from .ep import ExpertParallel
    for count_policy in ['collective','replicated']:
        for detail in ['fine','coarse']:
            captured=[]
            def capture(source,filename,*args,**kwargs):
                if filename=='<timing-only-EP-host-regions>':captured.append(copy.deepcopy(source))
                return real_compile(source,filename,*args,**kwargs)
            ep=SimpleNamespace(_run=lambda:None,count_policy=count_policy,_replicated_run=ExpertParallel._replicated_run)
            with patch('builtins.compile',capture):instrument_ep(ep,Markers(),detail=detail)
            assert len(captured)==1
            restored=RemoveObserver(remove_alias=count_policy=='collective').visit(captured[0])
            expected=original_dump if count_policy=='collective' else ast.dump(ExpertParallel._replicated_run._ep_observer_ast,include_attributes=False)
            assert ast.dump(restored,include_attributes=False)==expected,(detail,count_policy)
            checked.append(dict(detail=detail,count_policy=count_policy,tensor_expression_preserved=True))
    result=dict(status='PASS',checks=checked,evidence='CPU_AST_EXPRESSION_IDENTITY_CHECK',
        scope='Remove observer regions, count retention, and source-selection aliases. Compare the original static expression.',
        limitation='No GPU execution, numerical fidelity, observer overhead, or model quality is established here.')
    write(ROOT/'quality/e2e_attribution_ast_check.json',result)
    print(result)

if __name__=='__main__':main()
