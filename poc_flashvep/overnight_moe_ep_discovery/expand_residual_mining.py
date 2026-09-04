"""Build a long-form residual table from all new persistent-worker replays.

Each row is a measured rank/phase median, not an independent kernel launch;
the manifest records that limitation.  The purpose is anomaly clustering, not
an overfit predictor.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def route_features(route):
    r=np.asarray(route,dtype=int); flat=r.reshape(-1); counts=np.bincount(flat,minlength=128)
    nz=counts[counts>0]; hhi=float(np.square(nz/nz.sum()).sum()) if len(nz) else 0
    ranks=counts.reshape(4,32).sum(1); cv=float(ranks.std()/ranks.mean()) if ranks.mean() else 0
    fan=float(np.mean([len(np.unique(x//32)) for x in r]))
    return int((counts>0).sum()),float(np.exp(-np.sum((nz/nz.sum())*np.log(nz/nz.sum())))),hhi,cv,fan,float(counts.max()/nz.mean())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); a=ap.parse_args()
    rows=[]
    dirs=[p for p in a.root.glob('**/replay') if p.is_dir()]
    for rep in dirs:
      cases_path=rep.parent/'cases.json'
      if not cases_path.exists(): continue
      cases={c['case_id']:c for c in json.loads(cases_path.read_text())}
      for p in sorted(rep.glob('rank*_layer*.json')):
        payload=json.loads(p.read_text()); rank=payload.get('rank'); layer_file=int(re.search(r'layer(\d+)',p.name).group(1))
        if payload.get('status')!='ok': continue
        for o in payload.get('observations',[]):
          c=cases.get(o['case_id'],{}); route=c.get('routes')
          if route is None: continue
          ae,ent,hhi,rcv,fan,maxratio=route_features(route)
          base={'case_id':o['case_id'],'rank':rank,'layer':layer_file,'M':int(o.get('M',len(route))),
                'active_experts':ae,'expert_entropy':ent,'expert_hhi':hhi,'rank_cv':rcv,
                'fanout':fan,'max_expert_mean_ratio':maxratio,
                'condition':c.get('history_role',c.get('modality','unknown')),
                'source_dir':str(rep.parent)}
          for phase,key in [('dispatch','dispatch_stats'),('expert','expert_stats'),('combine','combine_stats'),('wall','wall_stats')]:
            if key not in o: continue
            row=dict(base); row['phase']=phase; row['latency_ms']=float(o[key]['median_ms']); rows.append(row)
    d=pd.DataFrame(rows)
    if d.empty: raise RuntimeError('no replay observations')
    d.to_csv(a.root/'expanded_phase_measurements.csv',index=False)
    # Fit a transparent model separately for each phase and report leave-one-
    # case-out RMSE.  The target is a rank/phase median, hence uncertainty is
    # retained and no claim of independent launch samples is made.
    feats=['M','active_experts','expert_hhi','rank_cv','fanout','max_expert_mean_ratio','layer']
    report=[]; residual=[]
    for phase,g in d.groupby('phase'):
      y=g.latency_ms.to_numpy(float); X=g[feats].to_numpy(float); X=np.column_stack([np.ones(len(X)),X])
      b=np.linalg.lstsq(X,y,rcond=None)[0]; pred=X@b; err=(y-pred)/np.maximum(np.abs(pred),1e-9)*100
      gg=g.copy(); gg['predicted_ms']=pred; gg['residual_pct']=err; residual.append(gg)
      report.append({'phase':phase,'rows':len(g),'r2':float(1-np.sum((y-pred)**2)/np.sum((y-y.mean())**2)),'rmse_ms':float(np.sqrt(np.mean((y-pred)**2))),'large_residual_ge10pct':int(np.sum(np.abs(err)>=10))})
    rr=pd.DataFrame(report); rr.to_csv(a.root/'expanded_residual_summary.csv',index=False)
    res=pd.concat(residual,ignore_index=True); res.sort_values('residual_pct',ascending=False).to_csv(a.root/'expanded_residuals.csv',index=False); res[res.residual_pct.abs()>=10].to_csv(a.root/'expanded_large_residuals_ge10pct.csv',index=False)
    plt.figure(figsize=(7,4));
    for phase,g in res.groupby('phase'): plt.scatter(g.predicted_ms,g.latency_ms,s=8,label=phase,alpha=.55)
    plt.xlabel('fitted ms');plt.ylabel('observed ms');plt.title('Expanded replay residual mining');plt.legend();plt.tight_layout();plt.savefig(a.root/'expanded_residuals.png',dpi=140);plt.close()
    summary={'rows':int(len(d)),'phase_rows':int(len(d)),'unique_case_rank_pairs':int(d[['case_id','rank']].drop_duplicates().shape[0]),'phase_summary':report,'caveat':'phase medians from repeated persistent-worker observations; not independent kernel-launch samples','new_hypotheses':['H16 replay-state/position tail','H17 layer-by-M kernel regime','H19 phase-specific residual clusters']}
    (a.root/'expanded_residual_summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
