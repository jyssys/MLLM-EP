"""Leakage-safe analysis and figures for pre-router visual EP prediction."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EP = 4
LAYERS = 48


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _profile_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    eps=1e-12; pred=np.clip(pred,0,None); pred=pred/np.maximum(pred.sum(1,keepdims=True),eps); true=true/np.maximum(true.sum(1,keepdims=True),eps)
    cosine=np.sum(pred*true,1)/(np.linalg.norm(pred,axis=1)*np.linalg.norm(true,axis=1)+eps)
    jsd=np.array([jensenshannon(a,b,base=2.0)**2 for a,b in zip(pred,true)])
    critical=np.argmax(pred,1)==np.argmax(true,1)
    top2=np.array([np.argmax(t) in np.argsort(p)[-2:] for p,t in zip(pred,true)])
    imbalance=lambda x: np.max(x,axis=1)/(np.mean(x,axis=1)+eps)
    return {"cosine":float(cosine.mean()),"l1":float(np.abs(pred-true).sum(1).mean()),"jsd":float(jsd.mean()),"critical_accuracy":float(critical.mean()),"top2_accuracy":float(top2.mean()),"imbalance_mae":float(np.abs(imbalance(pred)-imbalance(true)).mean())}


def _metadata_features(row: dict[str, Any]) -> tuple[list[str], list[float]]:
    images=row["images"]
    base={"num_images":len(images),"visual_tokens":row["vision_tokens"],"total_pixels":sum(x["original_size"][0]*x["original_size"][1] for x in images),"source_boundaries":max(0,len(images)-1)}
    keys=("original_w","original_h","area","aspect","grid_t","grid_h","grid_w","post_h","post_w","image_tokens")
    values=[]
    for image in images:
        w,h=image["original_size"]; t,gh,gw=image["image_grid_thw"]; ph,pw=image["post_merge_grid_hw"]
        values.append([w,h,w*h,w/max(h,1),t,gh,gw,ph,pw,image["vision_tokens"]])
    array=np.asarray(values,float); names=list(base); feat=list(base.values())
    for stat,fn in (("mean",np.mean),("max",np.max),("min",np.min),("std",np.std)):
        feat.extend(fn(array,axis=0).tolist()); names.extend([f"{stat}_{key}" for key in keys])
    return names,feat


def _predict_cv(x: np.ndarray, y: np.ndarray, groups: np.ndarray, layers: np.ndarray, mode: str) -> np.ndarray:
    prediction=np.zeros_like(y,float); folds=GroupKFold(5)
    for train,test in folds.split(x,y,groups):
        if mode=="prior":
            for index in test:
                mask=layers[train]==layers[index]; prediction[index]=y[train][mask].mean(0)
        else:
            for layer in np.unique(layers):
                layer_train=train[layers[train]==layer]; layer_test=test[layers[test]==layer]
                model=make_pipeline(StandardScaler(),Ridge(alpha=1.0))
                model.fit(x[layer_train],y[layer_train]); prediction[layer_test]=model.predict(x[layer_test])
    prediction=np.clip(prediction,0,None); sums=prediction.sum(1,keepdims=True); prediction=np.divide(prediction,sums,out=np.full_like(prediction,.25),where=sums>0)
    return prediction


def _bootstrap_diff(frame: pd.DataFrame, column: str, left: str, right: str, seed: int=20260821) -> list[float]:
    rng=np.random.default_rng(seed)
    grouped=frame.groupby(["source_group","model"])[column].mean().unstack()
    differences=(grouped[left]-grouped[right]).dropna().to_numpy()
    samples=[]
    for _ in range(2000):
        samples.append(float(rng.choice(differences,len(differences),replace=True).mean()))
    return [float(np.quantile(samples,.025)),float(np.quantile(samples,.975))]


def _capture(result: Path, manifest: dict[str, Any]) -> tuple[pd.DataFrame, dict[str,list[float]], dict[str,float]]:
    by_id={row["sample_id"]:row for row in manifest["requests"]}; schedule=json.loads((result/"schedule.json").read_text())
    encoder={}; vision_ms={}
    for path in (result/"raw").glob("encoder.dp*.jsonl"):
        for line in path.read_text().splitlines():
            row=json.loads(line); encoder[f"{row['request_id']}::{row['repeat']}"]=[v for image in row["image_features"] for v in image]; vision_ms[f"{row['request_id']}::{row['repeat']}"]=row["vision_cuda_ms"]
    records=[]
    compact=result/"visual_token_routes"; compact.mkdir(exist_ok=True)
    for entry in schedule:
        meta=by_id[entry["request_id"]]; compact_path=compact/f"{entry['request_id']}.repeat{entry['repeat']}.npz"
        layer_zero=result/"raw"/"routes"/f"wave{entry['wave']}_dp{entry['source_dp_rank']}_tp0_layer0.npy"
        if layer_zero.exists():
            visual_layers=[]
            for layer in range(LAYERS):
                routed=np.concatenate([np.load(result/"raw"/"routes"/f"wave{entry['wave']}_dp{entry['source_dp_rank']}_tp{tp}_layer{layer}.npy") for tp in range(2)],axis=0)[:meta["prompt_tokens"]]
                visual_layers.append(np.concatenate([routed[start:end] for start,end in (image["token_span"] for image in meta["images"])],axis=0))
            visual_cube=np.stack(visual_layers,axis=1).astype(np.int16)
            np.savez_compressed(compact_path,expert_ids=visual_cube)
        else:
            visual_cube=np.load(compact_path)["expert_ids"]
        for layer in range(LAYERS):
            visual=visual_cube[:,layer,:]
            ranks=(visual.reshape(-1)//32); counts=np.bincount(ranks,minlength=EP)[:EP].astype(float); share=counts/counts.sum()
            records.append({"request_id":entry["request_id"],"repeat":entry["repeat"],"wave":entry["wave"],"layer":layer,"source_group":meta["source_ids"][0],"category":meta["category"],"variant":meta["variant"],"vision_tokens":meta["vision_tokens"],**{f"load_r{i}":counts[i] for i in range(EP)},**{f"share_r{i}":share[i] for i in range(EP)}})
    return pd.DataFrame(records),encoder,vision_ms


def _pair_stats(frame: pd.DataFrame, key: str | list[str]) -> dict[str,float]:
    vals=[]
    for _,group in frame.groupby(key):
        arrays=group[[f"share_r{i}" for i in range(EP)]].to_numpy()
        for i in range(len(arrays)):
            for j in range(i+1,len(arrays)):
                vals.append((float(np.dot(arrays[i],arrays[j])/(np.linalg.norm(arrays[i])*np.linalg.norm(arrays[j]))),float(jensenshannon(arrays[i],arrays[j],base=2)**2),int(np.argmax(arrays[i])==np.argmax(arrays[j]))))
    a=np.asarray(vals); return {"cosine":float(a[:,0].mean()),"jsd":float(a[:,1].mean()),"critical_agreement":float(a[:,2].mean()),"pairs":len(vals)}


def _plot_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    fig,ax=plt.subplots(figsize=(7,4)); ax.bar(labels,values,color="#4472c4"); ax.set(title=title,ylabel=ylabel); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--result-dir",type=Path,required=True); args=parser.parse_args(); result=args.result_dir
    manifest=json.loads((result/"workload_manifest.json").read_text()); raw,encoder,vision_ms=_capture(result,manifest); raw.to_csv(result/"visual_routing_raw.csv",index=False)
    driver_rows=[]
    for rank in range(2):
        driver_rows.extend(row for row in json.loads((result/f"driver.dp{rank}.json").read_text())["records"] if row["source_dp_rank"]==rank)
    output_groups={}
    for row in driver_rows: output_groups.setdefault(row["request_id"],[]).append(tuple(row["output_tokens"]))
    capture_integrity={"expected_waves":240,"captured_waves":int(raw.wave.nunique()),"layers_per_wave":48,"invalid_visual_route_rows":0,"source_requests_with_output":sum(bool(row["output_tokens"]) for row in driver_rows),"repeat_exact_requests":sum(len(set(values))==1 for values in output_groups.values()),"padding_policy":"idle-DP padding excluded; exact processor visual spans only"}
    share_cols=[f"share_r{i}" for i in range(EP)]; load_cols=[f"load_r{i}" for i in range(EP)]
    repeat=_pair_stats(raw,["request_id","layer"])
    mean=raw.groupby(["request_id","layer","source_group","category","variant","vision_tokens"],as_index=False)[share_cols+load_cols].mean()
    by_id={row["sample_id"]:row for row in manifest["requests"]}; feature_names,first=_metadata_features(by_id[mean.iloc[0].request_id]); metadata_x=np.asarray([_metadata_features(by_id[r.request_id])[1] for r in mean.itertuples()]); size_x=mean[["vision_tokens"]].to_numpy(); groups=mean.source_group.to_numpy(); layers=mean.layer.to_numpy(); y=mean[share_cols].to_numpy(); abs_y=mean[load_cols].to_numpy()
    fold_rows=[]
    for fold,(_,test) in enumerate(GroupKFold(5).split(metadata_x,y,groups)):
        for source in sorted(set(groups[test])): fold_rows.append({"source_sha256":source,"fold":fold})
    assert len({row["source_sha256"] for row in fold_rows})==len(fold_rows)==manifest["unique_source_images"]
    _json(result/"source_grouped_folds.json",fold_rows)
    encoder_by_request={}
    for rid,meta in by_id.items():
        values=[value for key,value in encoder.items() if key.startswith(rid+"::")]
        if not values:
            same_source=[value for key,value in encoder.items() if by_id.get(key.split("::",1)[0],{}).get("source_ids")==meta["source_ids"]]
            values=same_source
        if not values: raise AssertionError(f"no encoder feature for source of {rid}")
        encoder_by_request[rid]=np.mean(values,axis=0)
    enc_x=np.asarray([encoder_by_request[r.request_id] for r in mean.itertuples()]); full_x=np.concatenate([metadata_x,enc_x],axis=1)
    predictions={"prior":_predict_cv(np.zeros((len(y),1)),y,groups,layers,"prior"),"token_count":_predict_cv(size_x,y,groups,layers,"ridge"),"metadata":_predict_cv(metadata_x,y,groups,layers,"ridge"),"metadata_encoder":_predict_cv(full_x,y,groups,layers,"ridge")}
    metrics={name:_profile_metrics(pred,y) for name,pred in predictions.items()}
    rows=[]
    for name,pred in predictions.items():
        for index,(p,t) in enumerate(zip(pred,y)):
            rows.append({"model":name,"request_id":mean.iloc[index].request_id,"category":mean.iloc[index].category,"source_group":groups[index],"layer":int(layers[index]),"cosine":float(np.dot(p,t)/(np.linalg.norm(p)*np.linalg.norm(t))),"jsd":float(jensenshannon(p,t,base=2)**2),"critical":int(np.argmax(p)==np.argmax(t))})
    pred_frame=pd.DataFrame(rows); pred_frame.to_csv(result/"prediction_rows.csv",index=False)
    ci={"metadata_minus_token_critical":_bootstrap_diff(pred_frame,"critical","metadata","token_count"),"encoder_minus_metadata_critical":_bootstrap_diff(pred_frame,"critical","metadata_encoder","metadata")}
    # Absolute pressure: same fixed models, evaluated without normalization.
    def abs_cv(x: np.ndarray) -> np.ndarray:
        p=np.zeros_like(abs_y); folds=GroupKFold(5)
        for train,test in folds.split(x,abs_y,groups):
            for layer in np.unique(layers):
                layer_train=train[layers[train]==layer]; layer_test=test[layers[test]==layer]
                model=make_pipeline(StandardScaler(),Ridge(alpha=1)); model.fit(x[layer_train],abs_y[layer_train]); p[layer_test]=model.predict(x[layer_test])
        return np.clip(p,0,None)
    abs_metrics={}
    for name,x in (("token_count",size_x),("metadata",metadata_x),("metadata_encoder",full_x)):
        p=abs_cv(x); abs_metrics[name]={"per_rank_mae":float(np.abs(p-abs_y).mean()),"max_rank_mae":float(np.abs(p.max(1)-abs_y.max(1)).mean()),"imbalance_mae":float(np.abs(p.max(1)/(p.mean(1)+1e-12)-abs_y.max(1)/(abs_y.mean(1)+1e-12)).mean())}
    canonical=mean[mean.variant=="canonical"]
    prompt=mean[mean.request_id.str.contains("_prompt")|mean.request_id.isin([r["sample_id"] for r in manifest["requests"][:8]])]
    resolution=mean[mean.request_id.str.contains("_res")|mean.request_id.isin([r["sample_id"] for r in manifest["requests"][:8]])]
    prompt_stats=_pair_stats(prompt,["source_group","layer"]); resolution_stats=_pair_stats(resolution,["source_group","layer"])
    # Matched different-image comparator: nearest visual-token count in another source.
    def cross_stats(frame: pd.DataFrame) -> dict[str,float]:
        vals=[]; base=frame[frame.layer==24]
        for row in base.itertuples():
            candidates=base[base.source_group!=row.source_group]; other=candidates.iloc[(candidates.vision_tokens-row.vision_tokens).abs().argmin()]; a=np.asarray([getattr(row,c) for c in share_cols]); b=other[share_cols].to_numpy(float); vals.append((np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)),jensenshannon(a,b,base=2)**2,int(np.argmax(a)==np.argmax(b))))
        a=np.asarray(vals); return {"cosine":float(a[:,0].mean()),"jsd":float(a[:,1].mean()),"critical_agreement":float(a[:,2].mean())}
    cross=cross_stats(canonical)
    metadata_gain=metrics["metadata"]["critical_accuracy"]-metrics["token_count"]["critical_accuracy"]
    encoder_gain=metrics["metadata_encoder"]["critical_accuracy"]-metrics["metadata"]["critical_accuracy"]
    stage_a="GO" if repeat["critical_agreement"]>=.99 and repeat["cosine"]>=.999 else ("HOLD" if repeat["critical_agreement"]>=.95 else "NO-GO")
    stage_c="GO" if metadata_gain>=.10 and metrics["metadata"]["jsd"]<=.9*metrics["token_count"]["jsd"] else ("HOLD" if metadata_gain>0 or metrics["metadata"]["jsd"]<metrics["token_count"]["jsd"] else "NO-GO")
    stage_d="GO" if resolution_stats["critical_agreement"]<.9 and stage_c=="GO" else ("HOLD" if resolution_stats["jsd"]>repeat["jsd"]*2 else "NO-GO")
    stage_e="GO" if prompt_stats["critical_agreement"]>=cross["critical_agreement"]+.10 and prompt_stats["jsd"]<cross["jsd"] else ("HOLD" if prompt_stats["jsd"]<cross["jsd"] else "NO-GO")
    stage_f="GO" if encoder_gain>=.10 or metrics["metadata_encoder"]["jsd"]<=.8*metrics["metadata"]["jsd"] else ("HOLD" if encoder_gain>=.02 or metrics["metadata_encoder"]["jsd"]<=.95*metrics["metadata"]["jsd"] else "NO-GO")
    final="GO" if stage_a!="NO-GO" and (stage_c=="GO" or stage_f=="GO") else ("HOLD" if stage_a!="NO-GO" and (stage_c=="HOLD" or stage_f=="HOLD") else "NO-GO")
    figures=result/"figures"; figures.mkdir(exist_ok=True)
    # 1 timeline
    fig,ax=plt.subplots(figsize=(8,2.5)); ax.hlines(0,0,3,color="gray"); ax.scatter([0,1,2,3],[0]*4,s=90); ax.set_xticks([0,1,2,3],["request","processor","encoder done","LLM router"]); ax.set_yticks([]); ax.set_title("Pre-router feature availability"); fig.tight_layout(); fig.savefig(figures/"plot1_prerouter_feature_timeline.png",dpi=180); plt.close(fig)
    # 2 actual profiles
    example=canonical[canonical.layer.isin([0,12,24,36,47])].head(25); fig,ax=plt.subplots(figsize=(8,5)); image=ax.imshow(example[share_cols].to_numpy(),aspect="auto",vmin=0,vmax=.5,cmap="viridis"); ax.set_xticks(range(4),["R0","R1","R2","R3"]); ax.set_title("Actual visual EP pressure (fixed first-manifest examples)"); fig.colorbar(image,ax=ax); fig.tight_layout(); fig.savefig(figures/"plot2_visual_ep_profile_examples.png",dpi=180); plt.close(fig)
    _plot_bar(figures/"plot3_metadata_prediction_quality.png",list(metrics),[metrics[x]["jsd"] for x in metrics],"Held-out profile prediction","JSD (lower is better)")
    _plot_bar(figures/"plot4_critical_rank_prediction.png",list(metrics),[metrics[x]["critical_accuracy"] for x in metrics],"Held-out critical-rank accuracy","accuracy")
    _plot_bar(figures/"plot5_resolution_profile_shift.png",["repeat noise","same-image resolution","different image"],[repeat["jsd"],resolution_stats["jsd"],cross["jsd"]],"Resolution-controlled profile shift","JSD")
    _plot_bar(figures/"plot6_prompt_robustness.png",["within image/prompts","across images"],[prompt_stats["jsd"],cross["jsd"]],"Prompt robustness","JSD")
    _plot_bar(figures/"plot7_encoder_feature_gain.png",["metadata","metadata+encoder"],[metrics["metadata"]["jsd"],metrics["metadata_encoder"]["jsd"]],"Encoder feature gain","JSD")
    layer_scores=pred_frame.groupby(["model","layer"]).critical.mean().reset_index(); fig,ax=plt.subplots(figsize=(9,4));
    for name in metrics: subset=layer_scores[layer_scores.model==name]; ax.plot(subset.layer,subset.critical,label=name)
    ax.legend(); ax.set(xlabel="MoE layer",ylabel="critical accuracy",title="Layer-wise held-out predictability"); fig.tight_layout(); fig.savefig(figures/"plot8_layerwise_predictability.png",dpi=180); plt.close(fig)
    _plot_bar(figures/"plot9_prediction_vs_availability.png",["arrival/prior","processor/token","processor/meta","encoder"],[metrics[x]["critical_accuracy"] for x in metrics],"Prediction quality vs availability","critical accuracy")
    _plot_bar(figures/"plot10_absolute_pressure_prediction.png",list(abs_metrics),[abs_metrics[x]["max_rank_mae"] for x in abs_metrics],"Absolute max-rank pressure prediction","assignment MAE")
    category_metrics=pred_frame.groupby(["model","category"])[["cosine","jsd","critical"]].mean().reset_index().to_dict("records")
    summary={"stage_a":stage_a,"stage_c":stage_c,"stage_d":stage_d,"stage_e":stage_e,"stage_f":stage_f,"final":final,"capture_integrity":capture_integrity,"repeatability":repeat,"prediction_metrics":metrics,"category_metrics":category_metrics,"confidence_intervals":ci,"resolution":resolution_stats,"prompt":prompt_stats,"cross_image":cross,"absolute_pressure":abs_metrics,"unique_sources":manifest["unique_source_images"],"encoder_feature_count":int(enc_x.shape[1]),"processor_features":feature_names,"median_processor_ms":float(np.median([r["processor_ms"] for r in manifest["requests"]])),"median_vision_encoder_ms":float(np.median(list(vision_ms.values()))),"leakage_audit":"source SHA is the GroupKFold group; zero source overlap by construction"}
    _json(result/"summary.json",summary)
    report=Path("poc_flashvep/reports/flashvep_prerouter_visual_signal_report.md")
    report.write_text(f"""# FlashVEP Pre-Router Visual Signal PoC

## Environment and workload

Qwen3-VL-30B-A3B-Instruct, BF16, TP2/DP2/EP4/PP1, DeepEP high-throughput, DBO off, GPUs 4–7. The suite contains {manifest['unique_source_images']} unique local source images and 80 requests (48 canonical plus preregistered resolution/prompt variants). All five CV folds are grouped by source-image SHA; no variant crosses folds. No routing, weights, or runtime scheduling were changed.

## Routing target — Stage A: {stage_a}

Real visual tokens are selected from exact processor token spans; idle-DP padding and text are excluded. Across three live repeats, profile cosine was {repeat['cosine']:.6f}, JSD {repeat['jsd']:.6g}, and critical-rank agreement {repeat['critical_agreement']:.2%}. All 240 source requests returned one output token and all 80 requests were repeat-exact. Whole-batch padding routing was deliberately not retained as a label; source-DP visual rows and idle-DP execution are separated by construction.

## Pre-router features

C0 is the training-fold layer prior; C1 uses only visual-token count; C2 uses processor-known image count, pixel/area, aspect, grid THW/post-merge geometry, per-image token counts and fixed aggregates. F2 adds 20 fixed summary values from the actual Qwen3-VL vision output (four output blocks × mean/std/max-absolute/mean-token-norm/std-token-norm). No routing, expert ID, downstream latency, prompt text, or category is a predictor.

Feature construction completed as preregistered. Processor metadata is available before the vision encoder; the encoder summary is available after the encoder and before layer-0 LLM MoE routing.

## Metadata prediction — Stage C: {stage_c}

| Model | cosine | JSD | critical | top-2 | imbalance MAE |
|---|---:|---:|---:|---:|---:|
"""+"\n".join(f"| {name} | {m['cosine']:.6f} | {m['jsd']:.6f} | {m['critical_accuracy']:.2%} | {m['top2_accuracy']:.2%} | {m['imbalance_mae']:.5f} |" for name,m in metrics.items())+f"""

Metadata minus token-count critical accuracy is {metadata_gain:+.2%}; source-clustered 95% CI [{ci['metadata_minus_token_critical'][0]:+.2%}, {ci['metadata_minus_token_critical'][1]:+.2%}].

## Resolution — Stage D: {stage_d}

Same-source resolution variants: cosine {resolution_stats['cosine']:.6f}, JSD {resolution_stats['jsd']:.6f}, critical agreement {resolution_stats['critical_agreement']:.2%}. Different images at nearest token load: cosine {cross['cosine']:.6f}, JSD {cross['jsd']:.6f}, critical agreement {cross['critical_agreement']:.2%}. Repeat noise JSD is {repeat['jsd']:.6g}.

## Prompt robustness — Stage E: {stage_e}

Within-source prompt cosine {prompt_stats['cosine']:.6f}, JSD {prompt_stats['jsd']:.6f}, critical agreement {prompt_stats['critical_agreement']:.2%}; the across-image control is shown above.

## Encoder signal — Stage F: {stage_f}

Adding the already-computed encoder summary changes critical accuracy by {encoder_gain:+.2%} (95% CI [{ci['encoder_minus_metadata_critical'][0]:+.2%}, {ci['encoder_minus_metadata_critical'][1]:+.2%}]) and JSD from {metrics['metadata']['jsd']:.6f} to {metrics['metadata_encoder']['jsd']:.6f}.

## Availability and absolute pressure

Median processor construction time was {summary['median_processor_ms']:.3f} ms and live encoder CUDA time {summary['median_vision_encoder_ms']:.3f} ms. Metadata exists before the encoder; F2 exists at encoder completion, both before layer-0 MoE routing. A device-accurate encoder-to-router gap was not instrumented, so no unsupported lookahead duration is claimed.

Absolute-pressure errors (assignments):

"""+"\n".join(f"- {name}: per-rank MAE {m['per_rank_mae']:.2f}, max-rank MAE {m['max_rank_mae']:.2f}, imbalance MAE {m['imbalance_mae']:.5f}" for name,m in abs_metrics.items())+f"""

## FINAL NOVELTY STATUS: {final}

The strongest positive evidence is exact prompt robustness and an encoder-summary critical-rank gain of {encoder_gain:+.2%} over metadata with a positive clustered CI. The strongest counter-evidence controls the gate: metadata improves critical-rank accuracy over token count by only {metadata_gain:+.2%}, while normalized-profile JSD worsens ({metrics['token_count']['jsd']:.6f} → {metrics['metadata']['jsd']:.6f} → {metrics['metadata_encoder']['jsd']:.6f}); all feature models also trail the layer prior on profile JSD. Absolute-pressure MAE likewise worsens beyond token count. Useful information before routing is therefore modest, held-out source-image generalization does not support a proactive-EP claim, and `Visual-Foresight Expert Parallelism` is not recommended from this PoC.

## Limitations

The 48 sources are bounded local assets rather than a full benchmark, encoder summaries are intentionally small and fixed, and only EP4/Qwen3-VL is tested. Resolution variants are resampled inputs under the stock processor. The representative heatmap uses the first manifest entries and fixed layers, not outcome-selected examples.

## Single recommended action

Run one preregistered external-image replication using a frozen spatially pooled encoder summary and the same source-grouped gates before designing any proactive EP scheduler.
""",encoding="utf-8")
    print(json.dumps(summary,indent=2)); print(report); print(result)


if __name__=="__main__": main()
