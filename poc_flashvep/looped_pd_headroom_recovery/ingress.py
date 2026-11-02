"""Use native asynchronous ingress for the round-robin Mixed DP2 baseline."""
from uuid import uuid4


def enqueue(pool,rid,tokens,params):
    if rid in pool.assignment:raise ValueError('Duplicate replica-pool request ID')
    chosen=getattr(pool,'_recovery_next_replica',0)%len(pool.clients)
    pool._recovery_next_replica=chosen+1
    client=pool.clients[chosen]
    generation=uuid4().hex
    client.requests[rid]=generation
    client.generations[generation]=rid
    try:
        # StageWorker supports the native "add" command. Do not wait for a
        # synchronous control reply while another request executes Prefill.
        client.send('add',request_id=generation,tokens=list(tokens),params=params)
    except BaseException:
        client.requests.pop(rid,None)
        client.generations.pop(generation,None)
        raise
    pool.assignment[rid]=chosen


def install():
    from ..looped_pd_abc.replicas import ReplicaPool
    if getattr(ReplicaPool,'_recovery_openloop_ingress',False):return
    ReplicaPool._recovery_openloop_ingress=True
    ReplicaPool.add_request=enqueue
