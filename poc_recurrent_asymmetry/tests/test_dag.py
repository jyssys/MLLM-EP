import unittest
import numpy as np
from poc_recurrent_asymmetry.dag import micro_graph,ready_set,validate_order,isomorphism,schedule


class ExactDependencies(unittest.TestCase):
    def test_generic_unroll_has_identical_edges(self):
        self.assertTrue(isomorphism(2,depth=2,layers=2)['edge_sets_equal'])
        self.assertTrue(isomorphism(8)['edge_sets_equal'])

    def test_legal_ready_set_and_wrap(self):
        graph=micro_graph(2,depth=2,layers=2)
        self.assertEqual(ready_set(graph,set()),{(0,0,0,'Q'),(0,0,1,'Q')})
        self.assertEqual(graph[1,0,0,'Q'],{(0,1,0,'N')})
        self.assertIn((0,0,0,'K'),graph[0,0,1,'K'])
        with self.assertRaises(AssertionError):validate_order(graph,[(1,0,0,'Q')])

    def test_two_chunk_two_loop_two_layer_exact_attention(self):
        # Fixed float64 toy operators test graph legality, not pretrained model quality.
        x=np.array([[.2,-.5],[.7,.3]])
        w=np.array([[.8,.1],[-.2,.6]])
        def norm(a):return a/np.sqrt((a*a).mean(-1,keepdims=True)+1e-6)
        truth=x.copy()
        for r in range(2):
            for l in range(2):
                q=truth@w;k=truth@w.T;v=truth*.7
                scores=q@k.T/np.sqrt(2)
                scores[np.triu_indices(2,1)]=-np.inf
                scores=np.exp(scores-np.max(scores,axis=-1,keepdims=True))
                truth=truth+(scores/scores.sum(-1,keepdims=True))@v
                truth=truth+np.tanh(truth@w)*.1
            truth=norm(truth)
        graph=micro_graph(2,depth=2,layers=2);done=set();values={};order=[]
        while len(done)<len(graph):
            # Prefer deeper available work. Never consume an unpublished KV row.
            node=max(ready_set(graph,done));r,l,c,kind=node
            if kind=='Q':
                incoming=values[r,l-1,c,'O'] if l else (values[r-1,1,c,'N'] if r else x[c:c+1])
                values[node]=(incoming,incoming@w,incoming@w.T,incoming*.7)
            elif kind=='K':values[node]=values[r,l,c,'Q']
            elif kind=='O':
                h,q,_,_=values[r,l,c,'K']
                k=np.concatenate([values[r,l,j,'K'][2] for j in range(c+1)])
                v=np.concatenate([values[r,l,j,'K'][3] for j in range(c+1)])
                scores=q@k.T/np.sqrt(2);scores=np.exp(scores-scores.max())
                h=h+(scores/scores.sum())@v;values[node]=h+np.tanh(h@w)*.1
            else:values[node]=norm(values[r,l,c,'O'])
            done.add(node);order.append(node)
        self.assertTrue(validate_order(graph,order))
        actual=np.concatenate([values[1,1,c,'N'] for c in range(2)])
        np.testing.assert_allclose(actual,truth,rtol=1e-12,atol=1e-12)

    def test_scheduler_keeps_all_work_and_resource_order(self):
        costs={(r,g,c):1. for r in range(4) for g in range(4) for c in range(3)}
        elapsed,rows=schedule(costs,lambda t:t[1],comm_ms=.1)
        self.assertEqual(sum(r['kind']=='COMPUTE' for r in rows),48)
        self.assertGreater(elapsed,0)
        serial,_=schedule(costs,lambda t:0)
        self.assertEqual(serial,48.)


if __name__=='__main__':unittest.main()
