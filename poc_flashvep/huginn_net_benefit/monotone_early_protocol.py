"""A pinned optional protocol fork for immutable, shallow KV publications.

Only the two policy checks for an empty receive set change. Native allocation,
fragment counts, producer events, validity checks, hidden import, ACK, and source
retention remain unchanged. The derived source is an explicit audit artifact.
"""
import ast
import hashlib
import os
import time
from .common import ROOT, write


def early_receipt_admissible(connector, transfer, cut):
    if not transfer.received:
        return True
    proof = getattr(connector, 'net_immutable_receipts', {}).get(transfer.tid)
    return bool(proof is not None and type(cut) is int and
        proof['depth'] <= cut < connector.cfg['depth'] and
        not transfer.outgoing and not transfer.cancelled and not transfer.ready and
        transfer.expected is None and transfer.cut == connector.cfg['depth'] and
        transfer.request.phase.value == 'receiving' and proof['KV_only'] and
        proof['full_prompt'] == len(transfer.request.prompt_token_ids))


def derive(source):
    tree = ast.parse(source)
    changed = []
    class Rewrite(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name != 'command':
                return self.generic_visit(node)
            class Checks(ast.NodeTransformer):
                def visit_UnaryOp(self, child):
                    value = child.operand
                    if isinstance(child.op, ast.Not) and isinstance(value, ast.Attribute) and (
                            value.attr == 'received' and isinstance(value.value, ast.Name)
                            and value.value.id == 't'):
                        changed.append(child.lineno)
                        return ast.copy_location(ast.Call(func=ast.Name(
                            id='early_receipt_admissible', ctx=ast.Load()), args=[
                            ast.Name(id='self', ctx=ast.Load()), ast.Name(id='t', ctx=ast.Load()),
                            ast.Subscript(value=ast.Name(id='m', ctx=ast.Load()),
                                slice=ast.Constant('depth'), ctx=ast.Load())], keywords=[]), child)
                    return self.generic_visit(child)
            return Checks().visit(node)
    result = Rewrite().visit(tree)
    assert len(changed) == 2, 'The pinned parent admission checks changed'
    ast.fix_missing_locations(result)
    return ast.unparse(result)+'\n', changed


def install(cfg):
    assert cfg['intervention_method'] in ('dynamic_loop', 'dynamic_tile')
    assert cfg.get('progressive_kv') and cfg.get('native_loop_event_transfer')
    assert min(cfg['boundary_checkpoints']) >= max(cfg.get('early_transfer_depths', [16]))
    from pathlib import Path
    from ..completion_aware_loop_pd import runtime as parent
    from ..native_loop_2d_oracle import native_path
    from vllm_rlt.pd.engine import PDEngine
    source = Path(parent.__file__).read_text()
    derived, lines = derive(source)
    namespace = dict(parent.__dict__, early_receipt_admissible=early_receipt_admissible)
    exec(compile(derived, str(ROOT/'raw'/cfg['name']/'derived_early_protocol.py'), 'exec'), namespace)
    namespace['install'](cfg)
    folder = ROOT/'raw'/cfg['name']
    # The source is frozen in each run. The derived text is reproducible from
    # that exact snapshot and these two explicit policy-predicate replacements.
    (folder/f'derived_early_protocol_{os.getpid()}.py').write_text(derived)
    write(folder/f'early_protocol_derivation_{os.getpid()}.json', dict(
        original_SHA256=hashlib.sha256(source.encode()).hexdigest(),
        derived_SHA256=hashlib.sha256(derived.encode()).hexdigest(), replaced_lines=lines,
        scope='Optional process-local protocol fork. Native transport and model implementations are unchanged. This is not a deployed vLLM upstream extension.'))
    connector = native_path.ACTIVE_CONNECTOR
    command = connector.command
    message = PDEngine._message
    records = []

    def messaged(engine, peer, item):
        if item['kind'] != 'idle_immutable_KV_frontier':
            return message(engine, peer, item)
        transfer = engine.transfers.get(item['tid'])
        if transfer is None or transfer.cancelled:
            return
        assert peer.name == transfer.p, 'Only the current source can announce immutable KV'
        engine._send(transfer.d, item['kind'], **{k:v for k,v in item.items() if k != 'kind'})

    def commanded(owner, item):
        if item['kind'] == 'idle_immutable_KV_frontier':
            transfer = owner.transfers.get(item['tid'])
            if transfer is None or transfer.cancelled:
                return True
            assert not transfer.outgoing and not transfer.ready and transfer.expected is None
            assert transfer.request.phase.value == 'receiving'
            assert item['KV_only'] is True and item['full_prompt'] == len(transfer.request.prompt_token_ids)
            assert type(item['depth']) is int and 0 < item['depth'] < cfg['depth']
            previous = getattr(owner, 'net_immutable_receipts', {})
            assert item['depth'] >= previous.get(item['tid'], {'depth': 0})['depth']
            previous[item['tid']] = dict(depth=item['depth'], KV_only=True, full_prompt=item['full_prompt'])
            owner.net_immutable_receipts = previous
            records.append(dict(time=time.monotonic(), event='IMMUTABLE_KV_FRONTIER_RECEIVED',
                tid=item['tid'], depth=item['depth'], already_received_fragments=len(transfer.received)))
            return True
        transfer = owner.transfers.get(item.get('tid'))
        if item['kind'] in ('idle_offer', 'idle_commit') and transfer is not None and transfer.received:
            records.append(dict(time=time.monotonic(), event='EARLY_RECEIPT_POLICY_CHECK',
                tid=transfer.tid, kind=item['kind'], depth=item['depth'], received_fragments=len(transfer.received),
                immutable_frontier_valid=early_receipt_admissible(owner, transfer, item['depth'])))
        return command(owner, item)

    close = connector.close
    forget = connector._forget
    def forgotten(owner, transfer):
        value = forget(owner, transfer)
        getattr(owner, 'net_immutable_receipts', {}).pop(transfer.tid, None)
        return value
    def closed(owner):
        try:
            return close(owner)
        finally:
            write(folder/f'immutable_early_receipts_{os.getpid()}.json', dict(
                role=owner.engine.stage_config.role, records=records,
                scope='A received immutable KV prefix does not activate the request. All numbered fragments, final hidden state, Native commit, import validity, and ACK are still required.'))
    connector.command, connector.close, connector._forget = commanded, closed, forgotten
    PDEngine._message = messaged


def qualification_receipt(folder):
    import json
    begin = json.loads((folder/'measurement_begin.json').read_text())['time']
    checks = [row for path in folder.glob('immutable_early_receipts_*.json')
        for row in json.loads(path.read_text())['records']
        if row['event'] == 'EARLY_RECEIPT_POLICY_CHECK' and row.get('kind') == 'idle_commit'
        and row['time'] >= begin
        and row['immutable_frontier_valid'] and row['received_fragments'] > 0]
    commits = {row['tid'] for path in folder.glob('intervention_*.json')
        for row in json.loads(path.read_text()).get('records', [])
        if row['event'] == 'DESTINATION_COMMIT' and row['time'] >= begin}
    qualified = [row for row in checks if row['tid'] in commits]
    return dict(passed=bool(qualified), committed_after_immutable_receipt=len(qualified),
        scope='Actual destination commit after one or more immutable KV fragments arrived. Exact state and lifetime qualification remain separate required gates.')
