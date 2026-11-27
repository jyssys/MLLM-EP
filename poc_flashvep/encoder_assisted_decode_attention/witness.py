"""Full-model fixed-prefix and Natural Greedy correctness on the physical E/P/D layout."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import torch

from .cache import Arena, DecodePages, PAGE, Placement, partition
from .model import Checkpoint, TokenDecoder
from .peer_attention import AssistedAttention


@dataclass
class Replay:
    placement: Placement
    tokens: list


def step_kv(arena, placement, produced):
    position = placement.d_rows + produced - 1
    page, row = divmod(position, PAGE)
    return arena.cache[:, :, placement.d_pages[page], row].clone()


def differences(expected, actual):
    delta = expected.float() - actual.float()
    return {"relative_l2": float(delta.norm() / expected.float().norm().clamp_min(1e-12)),
            "max_abs": float(delta.abs().max()), "bitwise_equal": torch.equal(expected, actual)}


def decision(logits):
    scores = logits[0].float().clone()
    first = int(scores.argmax())
    best = float(scores[first])
    scores[first] = -torch.inf
    second = int(scores.argmax())
    runner_up = float(scores[second])
    return {"ids": [first, second], "logits": [best, runner_up], "margin": best - runner_up}


def rollout(items, decoder, pools, mode, reserve, operator, *, teacher=None, eos=None, graph=None, ratio=.5,
            stage_timings=None):
    local, remote = pools
    requests, imported, first = [], [], []
    for row in items:
        d_ids, e_ids = partition(row["length"], row["visual"], mode, ratio)
        d_pages = local.acquire(d_ids.numel() + reserve)
        e_pages = remote.acquire(e_ids.numel()) if e_ids.numel() else []
        transfers = [local.enqueue(row["prefix"], d_ids, d_pages)]
        if e_pages:
            transfers.append(remote.enqueue(row["prefix"], e_ids, e_pages))
        for move in transfers:
            local.wait(move)
        helper_name = "E" if remote.device.index == 0 else "P"
        for store, indices, pages, name in ((local,d_ids,d_pages,"D"),(remote,e_ids,e_pages,helper_name)):
            if not len(indices):
                continue
            slots = (torch.tensor(pages, device=store.device)[:,None] * PAGE
                     + torch.arange(PAGE, device=store.device)).flatten()[:len(indices)]
            exact = 0
            for i, layer in enumerate(row["prefix"].layers):
                for j, original in enumerate((layer.keys, layer.values)):
                    original = original[0].transpose(0,1).index_select(0, indices.to(original.device))
                    received = store.cache[i,j].flatten(0,1).index_select(0,slots)
                    exact += int(torch.equal(received, original.to(store.device)))
            imported.append({"rank": name, "expected_tensors": 2*decoder.n_layers, "exact_tensors": exact,
                             "prompt_rows": indices.numel()})
        m = Placement(d_pages,e_pages,d_ids.numel(),e_ids.numel(),1,mode)
        requests.append(Replay(m,[row["first"]]))
        first.append(row["first"])
    logits_all, generated = [], []
    with torch.cuda.device(decoder.dev):
        for i in range(reserve-1):
            previous = first if i == 0 else teacher[i-1] if teacher is not None else generated[-1]
            state = DecodePages(local,remote,requests,operator,decoder.heads,decoder.dim)
            if graph is not None:
                state.graph_batch = graph.prepare(state)
                if state.graph_batch is None:
                    raise RuntimeError("Graph witness did not use the intended backend")
            values, logits, timings = decoder(previous, [row["length"]+row["delta"]+i for row in items], state,
                                               instrument=stage_timings is not None)
            if stage_timings is not None:
                stage_timings.append(timings)
            generated.append(values)
            if eos is None:
                kv = torch.stack([step_kv(local,r.placement,len(r.tokens)) for r in requests])
                logits_all.append((logits, kv))
            for r, value in zip(requests,values,strict=True):
                r.tokens.append(value)
            if eos is not None and values[0] in eos:
                break
    for r in requests:
        local.release(r.placement.d_pages)
        if r.placement.helper_pages:
            remote.release(r.placement.helper_pages)
    return imported, generated, logits_all


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--mode",choices=("position","visual_first","visual_all","matched_visual_count"),default="position")
    p.add_argument("--steps",type=int,default=16)
    p.add_argument("--seed",type=int,default=11)
    p.add_argument("--natural-cap",type=int,default=96)
    p.add_argument("--natural-details",action="store_true",help="Replay the first natural divergence on a common teacher prefix")
    p.add_argument("--profile-stages",action="store_true",help="Collect diagnostic CUDA-event enclosures on unassisted Full28")
    p.add_argument("--deterministic",action="store_true")
    p.add_argument("--graph",action="store_true")
    p.add_argument("--graph-splits",type=int,default=0)
    p.add_argument("--helper-rank",type=int,choices=(0,1),default=0)
    p.add_argument("--ratio",type=float,default=.5)
    p.add_argument("--packed-vision",action="store_true")
    args=p.parse_args()
    if torch.cuda.device_count()!=3 or args.steps<16 or not 0. < args.ratio <= 1.:
        p.error("Use three devices, at least 16 fixed output tokens and a positive split ratio")
    torch.manual_seed(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    ck=Checkpoint(args.model,"qwen-vl")
    e,pr,d=map(torch.device,("cuda:0","cuda:1","cuda:2"))
    helper_device = torch.device("cuda",args.helper_rank)
    with torch.inference_mode():
        backend=ck.replica(pr,vision_device=e,packed_vision=args.packed_vision)
        dec=TokenDecoder(ck,ck.replica(d),d)
        rows=[x for x in json.loads(args.manifest.read_text())["requests"] if x.get("study_source","Main100")=="Main100"]
        rows.sort(key=lambda x:x["total_input_tokens"])
        id_selection=[0,len(rows)//4,len(rows)//2,len(rows)-1]
        cases,vision_comparison=[],[]
        for index in id_selection:
            row=rows[index]
            item=ck.preprocess(row,workload="realistic")
            encoder=backend.model.visual
            if args.packed_vision:
                slots=[block.attn for block in encoder.blocks]
                optimized=[layer.forward for layer in slots]
                for layer in slots:
                    layer.forward=layer._eada_default_forward
                stock=ck.encode(encoder,item.copy(),e)
                for layer,forward in zip(slots,optimized,strict=True):
                    layer.forward=forward
            unprojected=ck.encode(backend.model.visual,item,e)
            if args.packed_vision:
                vision_comparison.append({"request_id":row["request_id"],"shape":list(stock.shape),
                                          **differences(stock,unprojected)})
            torch.cuda.current_stream(e).synchronize()
            prefix, first=ck.prefill(backend,item,unprojected,pr,verify_rope=True)
            if args.packed_vision:
                _, stock_first=ck.prefill(backend,item,stock,pr)
                vision_comparison[-1].update({"stock_first_id":stock_first,"packed_first_id":first,
                                               "first_token_exact":stock_first==first})
            cases.append({"length":item["ids"].shape[-1],"visual":item["visual"],"prefix":prefix,
                          "delta":item["delta"],"first":first,"request_id":row["request_id"]})
        pools=[Arena(dev,dec.n_layers,dec.kv_heads,dec.dim,6) for dev in (d,helper_device)]
        graph=None
        if args.graph:
            from .paged_graph import PagedGraphPool
            with torch.cuda.device(d):
                helper_ceiling = 32768 if args.mode in ("visual_all","matched_visual_count") else max(
                    PAGE, 1 << (max(1,int(32768*args.ratio))-1).bit_length())
                graph=PagedGraphPool(*pools,heads=dec.heads,dim=dec.dim,max_batch=4,
                                     fixed_d_context=32768,fixed_helper_context=helper_ceiling,
                                     num_splits=args.graph_splits)
                graph.prewarm((1,4),include_local=True,include_split=True)
        results=[]
        for items in ([cases[0]],[cases[3]],cases):
            operator=AssistedAttention((len(items),dec.heads,dec.dim),decode_device=d,helper_device=helper_device)
            imports,base_ids,base=rollout(items,dec,pools,"local",args.steps,None)
            checks={"base_import":imports}
            for mode in ("local",args.mode):
                copy,ids,output=rollout(items,dec,pools,mode,args.steps,operator,teacher=base_ids,graph=graph,ratio=args.ratio)
                checks[mode]={"imports":copy,"forced_argmax_agreement":sum(
                    a==b for x,y in zip(base_ids,ids,strict=True) for a,b in zip(x,y,strict=True)),
                    "compared_tokens":len(items)*len(base_ids),
                    "logits":[differences(a[0],b[0]) for a,b in zip(base,output,strict=True)],
                    "generated_kv":[differences(a[1],b[1]) for a,b in zip(base,output,strict=True)]}
            if len(items)==1:
                def natural(mode):
                    if items[0]["first"] in ck.eos:
                        return [items[0]["first"]]
                    return [items[0]["first"]] + [x[0] for x in
                           rollout(items,dec,pools,mode,args.natural_cap,operator,eos=ck.eos,graph=graph,ratio=args.ratio)[1]]
                a,b=natural("local"),natural(args.mode)
                checks["natural"]={"local_ids":a,"assisted_ids":b,"exact":a==b,
                                   "left_eos":a[-1] in ck.eos,"right_eos":b[-1] in ck.eos}
                first_diff = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
                if args.natural_details and first_diff is not None:
                    teacher = [[token] for token in a[1:first_diff]]
                    prefix = {}
                    references = {}
                    for mode in ("local", args.mode):
                        _, ids, state = rollout(items,dec,pools,mode,first_diff+1,operator,
                                                teacher=teacher,graph=graph,ratio=args.ratio)
                        prefix[mode] = decision(state[-1][0])
                        references[mode] = state[-1]
                        if ids[-1] != [prefix[mode]["ids"][0]]:
                            raise RuntimeError("Standalone replay and stored logits selected different tokens")
                    left, right = references["local"], references[args.mode]
                    checks["natural"]["first_divergence"] = {
                        "token_index_zero_based": first_diff, "local_and_assisted": prefix,
                        "same_prefix_logits": differences(left[0], right[0]),
                        "same_prefix_generated_kv": differences(left[1], right[1]),
                    }
            results.append({"requests":[item["request_id"] for item in items],
                            "prompt_rows":[item["length"] for item in items], "checks":checks})
        profiles=[]
        if args.profile_stages:
            for items in [[row] for row in cases] + [cases]:
                timings=[]
                rollout(items,dec,pools,"local",args.steps,None,stage_timings=timings)
                profiles.append({"batch":len(items),"prompt_rows":[item["length"] for item in items],
                                 "decode_steps":timings})
        graph_receipt=graph.receipt() if graph is not None else None
        if graph is not None:
            graph.close()
    with args.out.open("x") as stream:
        json.dump({"evidence":"CORRECTNESS_ONLY","deterministic":args.deterministic,
                   "fixed_output_length":args.steps,"graph":graph_receipt,"ratio":args.ratio,
                   "helper_rank":args.helper_rank,
                   "packed_vision_vs_stock":vision_comparison,"results":results,
                   "diagnostic_full28_stage_event_profile":profiles},stream,indent=2)
        stream.write("\n")


if __name__=="__main__":
    main()
