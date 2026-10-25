"""Reversible native hooks and pre-GEMM edge omission/reuse experiments.

The disabled path calls the original implementation without a replacement.
The probe path returns each original tensor unchanged.
All caches reset at the beginning of every complete model forward.
"""
from collections import Counter
import math
import types
import hashlib
import torch

class Controller:
    def __init__(self, adapter, mode='probe', q=0., guard=None, renorm=False, oracle=None, matched_budget=None, random_scope='legacy_pass'):
        self.a, self.mode, self.q = adapter, mode, q
        self.guard, self.renorm, self.oracle = guard, renorm, oracle
        self.matched_budget = matched_budget
        self.random_scope = random_scope
        self.edge_counts = {}
        self.cache_sizes = {}
        self.handles, self.saved = [], []
        self.frames = []
        self.calls = Counter()
        self.router_state, self.previous = {}, {}
        self.raw_inputs, self.last_frame = {}, {}
        self.work = Counter()
        self.times = []
        self.sample = None
        self.current = None
        self.profile = mode == 'profile'

    def reset(self, length):
        self.calls.clear()
        self.router_state.clear()
        self.previous.clear()
        self.raw_inputs.clear()
        self.last_frame.clear()
        self.frames = []
        self.work.clear()
        self.edge_counts.clear()
        self.cache_sizes.clear()
        self.times = []
        if self.profile:
            self.sample=None
            return
        # Fixed positions are identical for all sites and passes in this forward.
        starts = torch.linspace(0,max(0,length-8),4,device=self.a.device).long()
        self.sample = (starts[:,None]+torch.arange(min(8,length),device=self.a.device)[None,:]).flatten().unique()

    def __enter__(self):
        self.a.controller = self
        if self.mode == 'disabled':
            return self
        for site, module in self.a.sites.items():
            if self.mode == 'probe':
                norm = module.norm if self.a.name.startswith('ifm_') else module.ln2
                def norm_pre(norm, inputs, site=site):
                    self.raw_inputs[site]=inputs[0]
                def block_post(block, inputs, output, site=site):
                    frame=self.last_frame.get(site)
                    if frame is not None:
                        frame['post_moe']=output[0][0,self.sample].detach().cpu()
                self.handles.append(norm.register_forward_pre_hook(norm_pre))
                self.handles.append(module.register_forward_hook(block_post))
            def router_hook(router, inputs, output, site=site):
                self.route(site, router, inputs[0], output)
            self.handles.append(module.router.register_forward_hook(router_hook))
            if self.a.name.startswith('ifm_'):
                original = module.experts.forward
                self.saved.append((module.experts, 'forward', original))
                def expert_forward(expert, x, counts, site=site, original=original):
                    return self.execute(site, x, counts, original)
                module.experts.forward = types.MethodType(expert_forward, module.experts)
        if self.a.name == 'foil1':
            import sys
            source = sys.modules[type(self.a.model).__module__]
            self.foil_gmm = source.grouped_mm
            self.saved.append((source, 'grouped_mm', self.foil_gmm))
            source.grouped_mm = self.foil_grouped_mm
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        for obj, name, original in reversed(self.saved):
            setattr(obj, name, original)
        self.a.controller = None

    def route(self, site, router, x, output):
        pass_index = self.calls[site]
        self.calls[site] += 1
        if self.a.name == 'foil1':
            logits, probs, weights, ids = output
            weights = weights.to(x.dtype)
        else:
            weights, ids, _, _ = output
            # This extra computation is diagnostic only. It never changes routing.
            if self.mode == 'probe':
                logits = torch.nn.functional.linear(x[:, self.sample], router.weight).float()
                scores = torch.sigmoid(logits) if router.score_func == 'sigmoid' else logits.softmax(-1)
                probs = scores / scores.sum(-1, keepdim=True)
            else:
                logits = probs = None
        self.router_state[site] = dict(ids=ids[0], weights=weights[0], x=x[0], pass_index=pass_index)
        self.current = site
        self.foil_phase = 0
        if self.mode == 'probe':
            p = probs[0, self.sample] if self.a.name == 'foil1' else probs[0]
            self.router_state[site]['entropy'] = -(p * p.clamp_min(1e-30).log()).sum(-1)
            top = p.topk(2, dim=-1).values
            self.router_state[site]['confidence'] = top[:, 0]
            self.router_state[site]['margin'] = top[:, 0]-top[:, 1]

    def begin(self, site, x, counts):
        state = self.router_state[site]
        ids, weights = state['ids'], state['weights']
        flat = ids.flatten()
        order = flat.argsort(stable=True)
        state['order'] = order
        state['counts'] = counts
        n = len(flat)
        self.work['total_edges'] += n
        state['keep'] = torch.ones(n, dtype=torch.bool, device=x.device)
        state['replacement'] = None
        recurrent = site in self.a.recurrent
        previous = self.previous.get(site)
        key = (site, state['pass_index'])
        if self.mode in ('disabled', 'probe', 'profile') or not recurrent or self.q == 0:
            self.edge_counts[key] = 0
            return state
        remove = torch.zeros_like(ids, dtype=torch.bool)
        replacement = torch.zeros((n, x.shape[-1]), dtype=x.dtype, device=x.device)
        if self.mode in ('oracle_drop', 'oracle_reuse'):
            ref = self.oracle[key]
            score = ref['contribution'] if self.mode == 'oracle_drop' else ref['reuse_error']
            eligible = torch.isfinite(score)
            count = min(int(n*self.q), int(eligible.sum()))
            if count:
                indices = score.flatten().topk(count, largest=False).indices
                remove.flatten()[indices] = True
            # A changed router must not cause an oracle slot to remove another expert.
            remove &= ids == ref['ids']
        elif self.mode == 'random_drop':
            # Per-site/pass row counts match the requested drop oracle count.
            count = self.matched_budget[key] if self.matched_budget is not None else int(n*self.q)
            site_seed=int(hashlib.sha256(site.encode()).hexdigest()[:8],16) if self.random_scope=='independent_site' else 0
            gen = torch.Generator(device=x.device).manual_seed(20261026 + state['pass_index'] + site_seed)
            remove.flatten()[torch.randperm(n, device=x.device, generator=gen)[:count]] = True
        elif self.mode == 'causal_drop':
            remove = weights < self.q
        elif self.mode == 'reuse':
            remove.fill_(True)
        else:
            raise ValueError(self.mode)
        if self.mode in ('reuse', 'oracle_reuse'):
            if previous is None:
                remove.zero_()
            else:
                match = ids[:, :, None] == previous['ids'][:, None, :]
                exists = match.any(-1)
                slot = match.long().argmax(-1)
                oldout = previous['outputs'].gather(1, slot[..., None].expand(-1,-1,x.shape[-1]))
                if self.guard is not None:
                    delta = (state['x'].float()-previous['x'].float()).norm(dim=-1)
                    relative = delta / state['x'].float().norm(dim=-1).clamp_min(1e-8)
                    exists &= relative[:, None] <= self.guard
                remove &= exists
                replacement = oldout.reshape(n, -1)[order]
        state['remove'] = remove
        state['keep'] = ~remove.flatten()[order]
        state['replacement'] = replacement
        if self.renorm and self.mode in ('causal_drop', 'oracle_drop', 'random_drop'):
            surviving = weights.masked_fill(remove, 0).sum(-1).clamp_min(1e-8)
            # Original selected weights sum to one. Apply this scale before native combine.
            state['renorm_scale'] = (1/surviving).repeat_interleave(ids.shape[-1])[order]
        self.work['omitted_edges'] += int(remove.sum())
        self.edge_counts[key] = int(remove.sum())
        return state

    def finish(self, site, y):
        if self.mode not in ('probe', 'reuse', 'oracle_reuse'):
            return y
        state = self.router_state[site]
        ids, weights, order = state['ids'], state['weights'], state['order']
        n, k = ids.shape
        inverse = order.argsort()
        outputs = y[inverse].view(n, k, -1)
        if self.mode in ('reuse', 'oracle_reuse'):
            self.previous[site] = dict(ids=ids.detach().clone(), outputs=outputs.detach().clone(), x=state['x'].detach().clone())
            self.cache_sizes[site]=sum(t.numel()*t.element_size() for t in self.previous[site].values())
            self.work['cache_peak_tensor_bytes']=max(self.work['cache_peak_tensor_bytes'],sum(self.cache_sizes.values()))
        if self.mode == 'probe':
            old = self.previous.get(site)
            reuse_error = torch.full_like(weights.float(), float('inf'))
            if old is not None:
                match = ids[:,:,None] == old['ids'][:,None,:]
                slot = match.long().argmax(-1)
                oldout = old['outputs'].gather(1,slot[...,None].expand(-1,-1,outputs.shape[-1]))
                reuse_error = (outputs.float()-oldout.float()).norm(dim=-1)*weights.float()
                reuse_error[~match.any(-1)] = float('inf')
            self.previous[site] = dict(ids=ids.detach().clone(), outputs=outputs.detach().clone())
            sample = self.sample
            self.frames.append(dict(site=site, recurrent=site in self.a.recurrent,
                pass_index=state['pass_index'], positions=sample.detach().cpu(),
                ids=ids[sample].detach().cpu(), weights=weights[sample].detach().cpu(),
                x=state['x'][sample].detach().cpu(), outputs=outputs[sample].detach().cpu(),
                raw_x=self.raw_inputs[site][0,sample].detach().cpu(),
                entropy=state['entropy'].detach().cpu(), confidence=state['confidence'].detach().cpu(),
                margin=state['margin'].detach().cpu(),
                full_ids=ids.detach().cpu(),
                full_reuse_error=reuse_error.detach().cpu(),
                full_contribution=(outputs.float().norm(dim=-1)*weights.float()).detach().cpu(),
                dtype=str(y.dtype), shape=list(outputs.shape)))
            self.last_frame[site]=self.frames[-1]
        return y

    def execute(self, site, x, counts, original):
        if self.profile:
            state = self.router_state[site]
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            y = original(x, counts)
            end.record()
            self.times.append((site, state['pass_index'], start, end))
            self.work['total_edges'] += x.shape[0]
            self.work['executed_edges'] += x.shape[0]
            return y
        state = self.begin(site, x, counts)
        keep = state['keep']
        if self.profile:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        if self.mode in ('probe', 'profile') or bool(keep.all()):
            y = original(x, counts)
            self.work['executed_edges'] += x.shape[0]
        else:
            group_ids = state['ids'].flatten()[state['order']]
            new_counts = torch.bincount(group_ids[keep], minlength=len(counts)).tolist()
            y = torch.zeros_like(x)
            # Only these rows enter the original expert GEMMs.
            if bool(keep.any()):
                y[keep] = original(x[keep], new_counts)
            self.work['executed_edges'] += int(keep.sum())
            if state['replacement'] is not None:
                y[~keep] = state['replacement'][~keep]
        if self.profile:
            end.record()
            self.times.append((site, state['pass_index'], start, end))
        if 'renorm_scale' in state:
            y = y * state['renorm_scale'][:, None].to(y.dtype)
        return self.finish(site, y)

    def foil_grouped_mm(self, x, weight, *, offs=None, **kwargs):
        site = self.current
        state = self.router_state[site]
        phase = self.foil_phase
        self.foil_phase += 1
        if self.profile:
            if phase == 0:
                state['start'] = torch.cuda.Event(enable_timing=True)
                state['start'].record()
            y = self.foil_gmm(x, weight, offs=offs, **kwargs)
            if phase == 2:
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                self.times.append((site, state['pass_index'], state['start'], end))
            return y
        if phase == 0:
            counts = torch.diff(offs, prepend=offs.new_zeros(1)).tolist()
            self.begin(site, x, counts)
            if self.profile:
                state['start'] = torch.cuda.Event(enable_timing=True)
                state['start'].record()
        keep = state['keep']
        selected = self.mode not in ('probe', 'profile') and not bool(keep.all())
        if selected:
            if phase < 2:
                x = x[keep]
            groups = state['ids'].flatten()[state['order']]
            new_counts = torch.bincount(groups[keep], minlength=len(offs))
            offs = new_counts.cumsum(0).int()
        if x.shape[0]:
            y = self.foil_gmm(x, weight, offs=offs, **kwargs)
        else:
            y = x.new_empty((0, weight.shape[-1]))
        if phase == 2:
            self.work['executed_edges'] += int(keep.sum()) if selected else keep.numel()
            if selected:
                full = y.new_zeros((keep.numel(), y.shape[-1]))
                full[keep] = y
                if state['replacement'] is not None:
                    full[~keep] = state['replacement'][~keep]
                y = full
            if self.profile:
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                self.times.append((site, state['pass_index'], state['start'], end))
            if 'renorm_scale' in state:
                y = y * state['renorm_scale'][:, None].to(y.dtype)
            self.finish(site, y)
        return y

def oracle_plan(frames):
    """Offline selection data. This function has access to baseline expert outputs."""
    previous, result = {}, {}
    for frame in frames:
        if not frame['recurrent']:
            continue
        key = (frame['site'], frame['pass_index'])
        ids, score = frame['full_ids'], frame['full_contribution']
        error = frame['full_reuse_error']
        result[key] = dict(ids=ids.cuda(), contribution=score.cuda(), reuse_error=error.cuda())
        previous[frame['site']] = frame
    return result
