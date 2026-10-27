"""Reuse TRUE EP. Add an exact, generic source-owner control, not a loop method."""
import ast
import copy
import inspect
import textwrap
import torch
from ..looped_moe_ep import runtime as original

def source_owner_function():
    tree=ast.parse(textwrap.dedent(inspect.getsource(original.PhysicalEP._run)))
    class ReplaceSourceRows(ast.NodeTransformer):
        def visit_Subscript(self,node):
            node=self.generic_visit(node)
            if isinstance(node.slice,ast.Slice) and ast.unparse(node.slice)=='self.rank::self.world':
                node.slice=ast.Name(id='source_rows',ctx=ast.Load())
            return node
    tree=ReplaceSourceRows().visit(tree)
    body=tree.body[0].body
    at=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='flat')
    body[at+1:at+1]=ast.parse('source_rows = self.source_rows(ids)').body
    scope=dict(vars(original))
    exec(compile(ast.fix_missing_locations(tree),'<generic-current-router-source-control>','exec'),scope)
    return scope['_run']

def replicated_count_function():
    """Use exact current routes already present on every dense-state rank."""
    tree=ast.parse(textwrap.dedent(inspect.getsource(original.PhysicalEP._run)))
    class SourceRows(ast.NodeTransformer):
        def visit_Subscript(self,node):
            node=self.generic_visit(node)
            if isinstance(node.slice,ast.Slice) and ast.unparse(node.slice)=='self.rank::self.world':
                node.slice=ast.Name(id='source_rows',ctx=ast.Load())
            return node
    tree=SourceRows().visit(tree);body=tree.body[0].body
    flat=next(i for i,n in enumerate(body) if original.assign_name(n)=='flat')
    body[flat+1:flat+1]=ast.parse("source_rows = slice(self.rank, None, self.world) if self.source_policy == 'static' else self.source_rows(ids)").body
    start=next(i for i,n in enumerate(body) if original.assign_name(n)=='counts')
    end=next(i for i,n in enumerate(body) if original.assign_name(n)=='matrix')
    body[start:end+1]=ast.parse('''
sources = torch.arange(b*l, device=ids.device) % self.world if self.source_policy == 'static' else ids.reshape(-1,k)[:,0] % self.world
matrix_tensor = torch.bincount((sources[:,None]*e + ids.reshape(-1,k)).reshape(-1), minlength=self.world*e).reshape(self.world,e)
matrix = matrix_tensor.tolist()
if self.verify_metadata:
    observed = torch.bincount(local_ids.flatten(), minlength=e).to(torch.int64)
    reference = torch.empty(self.world*e, dtype=torch.int64, device=hidden.device)
    dist.all_gather_into_tensor(reference, observed)
    expected_matrix = reference.reshape(self.world,e).tolist()
    assert matrix == expected_matrix, ('Replicated router counts differ', site, loop, self.rank, self.source_policy, matrix, expected_matrix)
''').body
    scope=dict(vars(original))
    exec(compile(ast.fix_missing_locations(tree),'<generic-exact-current-router-metadata>','exec'),scope)
    result=scope['_run'];result._ep_observer_ast=copy.deepcopy(tree)
    return result

class ExpertParallel(original.PhysicalEP):
    source_policy='static'
    count_policy='collective'
    verify_metadata=False
    _owner_run=source_owner_function()
    _replicated_run=replicated_count_function()

    def set_policy(self,name):
        assert name in ['static','source_owner','replicated_counts','replicated_counts_source_owner']
        self.source_policy='source_owner' if 'source_owner' in name else 'static'
        self.count_policy='replicated' if name.startswith('replicated_counts') else 'collective'

    def source_rows(self,ids):
        # Dense hidden states and current gate outputs are already replicated.
        # This selects a source among existing copies. It moves no KV or hidden state.
        sources=ids.reshape(-1,ids.shape[-1])[:,0]%self.world
        return torch.nonzero(sources==self.rank,as_tuple=True)[0]

    def _run(self,site,hidden,ids,weights):
        if self.count_policy=='replicated':return self._replicated_run(site,hidden,ids,weights)
        if self.source_policy=='static':return super()._run(site,hidden,ids,weights)
        assert self.source_policy=='source_owner'
        return self._owner_run(site,hidden,ids,weights)

