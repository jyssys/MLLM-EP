"""Pure offline metrics; independent unit is an entire trajectory."""
import numpy as np


def normalized(a):
    a=np.asarray(a,dtype=float)
    return a/np.maximum(a.sum(axis=-1,keepdims=True),1)


def cosine(a,b):
    return np.sum(a*b,axis=-1)/np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-30)


def jsd(a,b):
    p=normalized(a);q=normalized(b);m=(p+q)/2
    def kl(x):return np.sum(np.where(x>0,x*np.log2(np.maximum(x,1e-30)/np.maximum(m,1e-30)),0),axis=-1)
    return (kl(p)+kl(q))/2


def top(a,m):
    # Stable tie rule independent of next-turn labels.
    return np.argsort(-a,axis=-1,kind='stable')[...,:m]


def coverage(pred,truth,m):
    ids=top(pred,m)
    return np.take_along_axis(truth,ids,axis=-1).sum(axis=-1)/np.maximum(truth.sum(axis=-1),1)


def overlap(a,b,m):
    x=top(a,m);y=top(b,m)
    return np.array([len(set(i)&set(j))/m for i,j in zip(x,y)])


def clustered(values,clusters,seed=3407,iterations=2000):
    values=np.asarray(values,dtype=float);clusters=np.asarray(clusters)
    unique=sorted(set(clusters));means=np.array([values[clusters==x].mean() for x in unique])
    random=np.random.default_rng(seed)
    sampled=means[random.integers(0,len(means),(iterations,len(means)))].mean(axis=1)
    return dict(mean=float(means.mean()),median=float(np.median(values)),p25=float(np.quantile(values,.25)),
        p75=float(np.quantile(values,.75)),ci95=np.quantile(sampled,[.025,.975]).tolist(),
        trajectories=len(unique),pairs=len(values),bootstrap=iterations,independent_unit='trajectory',
        per_trajectory={x:float(v) for x,v in zip(unique,means)})


def traffic(ids,source,replicas=None):
    """Exact useful BF16 remote row attribution, destination dedup retained.

    ids[layer,token,k], source[token]; replica[layer,rank,expert]. No kernel or
    model execution occurs; selected experts/weights remain unchanged.
    """
    owners=ids.astype(np.int64)//32
    src=np.broadcast_to(np.asarray(source)[None,:,None],owners.shape)
    if replicas is not None:
        layer=np.arange(ids.shape[0])[:,None,None]
        hit=replicas[layer,src,ids]
        owners=np.where(hit,src,owners)
    remote=owners!=src
    remote_paths=int(remote.sum())
    rank_load=np.stack([(owners==r).sum(axis=(1,2)) for r in range(4)],axis=-1)
    matrix=np.zeros((ids.shape[0],4,4),dtype=np.int64)
    for s in range(4):
        mask=np.asarray(source)==s
        for destination in range(4):
            matrix[:,s,destination]=np.any(owners[:,mask,:]==destination,axis=-1).sum(axis=-1)
    diagonal=matrix[:,np.arange(4),np.arange(4)].sum()
    rows=int(matrix.sum()-diagonal)
    return dict(remote_paths=remote_paths,remote_dispatch_bytes=rows*4096,remote_combine_bytes=rows*4096,
        max_rank_load=float(rank_load.max(axis=-1).mean()),
        rank_max_mean=float((rank_load.max(axis=-1)/np.maximum(rank_load.mean(axis=-1),1)).mean()),
        rank_cv=float((rank_load.std(axis=-1)/np.maximum(rank_load.mean(axis=-1),1)).mean()),
        max_rank_remote_bytes=float(np.maximum(
            matrix.sum(axis=1)-matrix[:,np.arange(4),np.arange(4)],
            matrix.sum(axis=2)-matrix[:,np.arange(4),np.arange(4)]).max(axis=-1).mean()*4096),
        rank_load=rank_load.tolist(),matrix=matrix.tolist())


def source_counts(ids,source):
    counts=np.zeros((48,4,128),dtype=np.int64)
    for layer in range(48):
        for rank in range(4):counts[layer,rank]=np.bincount(ids[layer,np.asarray(source)==rank,:].ravel(),minlength=128)
    return counts


def choose_replicas(predicted,budget):
    result=np.zeros((48,4,128),dtype=bool)
    for layer in range(48):
        for rank in range(4):
            scores=predicted[layer,rank].copy()
            scores[rank*32:(rank+1)*32]=0
            selected=top(scores,budget)
            selected=selected[scores[selected]>0]
            result[layer,rank,selected]=True
    return result
