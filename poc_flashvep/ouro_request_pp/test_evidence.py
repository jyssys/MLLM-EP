"""Check published physical evidence without starting CUDA."""
import collections
import gzip
import json
import unittest
from .contract import ROOT,route


def load(path):
    with gzip.open(path,'rt') as f:return json.load(f)


class Evidence(unittest.TestCase):
    def test_final_correctness(self):
        files=list(ROOT.glob('correctness_checks_graph_v5*.json'))+list(ROOT.glob('correctness_checks_graph_receive_pool*.json'))
        self.assertTrue(files)
        for f in files:
            for r in json.loads(f.read_text()):
                self.assertTrue(r['PASS'])
                self.assertTrue(r['KV_bitwise'])
                self.assertEqual(r['logit_error']['max_abs'],0.)
                self.assertEqual(r['KV_slots'],24*r['depth'])

    def test_measured_routes_and_memory(self):
        files=list((ROOT/'raw').glob('primary_*_v5_r*/*.json.gz'))
        self.assertTrue(files)
        for f in files:
            d=load(f);count=d['C']*d['output_len'];path=route(d['placement'],d['depth'])
            self.assertEqual(d['graph_captures_during_run'],[0]*4,str(f))
            self.assertEqual(sum(v.get('hidden',0) for v in d['transfer_counts']),count*(len(path)-1),str(f))
            self.assertEqual(sum(v.get('token_return',0) for v in d['transfer_counts']),d['C']*(d['output_len']-1),str(f))
            for out in d['outputs']:
                self.assertEqual(len(out['ids']),d['output_len'])
                for rank,m in enumerate(d['memory']):
                    observed=m['caches'][out['id']]['slots']
                    expected=[24*loop+i for target,loop,start,stop in path if target==rank for i in range(start,stop)]
                    self.assertEqual(observed,sorted(expected),str(f))
                    for shape in m['caches'][out['id']]['shapes'].values():
                        self.assertEqual(shape,[1,16,len(out['input_ids'])+d['output_len']-1,128])

    def test_direct_placement_output_parity(self):
        groups=collections.defaultdict(dict)
        for f in (ROOT/'raw').glob('primary_*_v5_r*/*.json.gz'):
            d=load(f);c=d['case'];key=(c['depth'],c['C'],c['P'],c['O'],c['repeat'])
            groups[key][d['placement']]={r['id']:r['ids'] for r in d['outputs']}
        pairs=0
        for key,g in groups.items():
            if len(g)!=2:continue
            self.assertEqual(g['layer_pp4'],g['loop_pp4'],str(key));pairs+=1
        self.assertGreater(pairs,0)

    def test_actual_different_request_kernel_overlap(self):
        for arm in ['layer_pp4','loop_pp4']:
            f=ROOT/'profiles'/f'{arm}_R4_C4_v5'/'summary.json'
            self.assertTrue(f.exists(),str(f))
            d=json.loads(f.read_text())
            self.assertGreater(d['different_request_kernel_overlap_ms'],0)
            self.assertGreaterEqual(d['max_simultaneous_kernel_ranks'],2)
            self.assertEqual(d['tagged_kernels'],d['total_kernels'])
            raw=load(ROOT/'raw'/f'nsys_{arm}_R4_C4_v5'/'case_0000.json.gz')
            expected=sum(e['transfer_bytes'] for e in raw['events'] if e['transfer_kind'] in ['hidden','token_return'])
            peer=[x for x in d['copy_categories'] if x['copyKind']==10]
            self.assertEqual(sum(x['bytes'] for x in peer),expected)

    def test_common_backend_and_observer_output_parity(self):
        expected={}
        for f in (ROOT/'raw').glob('primary_*_v5_r0/*.json.gz'):
            d=load(f)
            expected[(d['placement'],d['depth'],d['C'],tuple(d['input_lengths']))]={r['id']:r['ids'] for r in d['outputs']}
        checked=0
        for directory in (ROOT/'raw').iterdir():
            if not directory.name.startswith(('receive_pool_','observer_','nsys_')):continue
            for f in directory.glob('*.json.gz'):
                d=load(f);key=(d['placement'],d['depth'],d['C'],tuple(d['input_lengths']))
                if key not in expected:continue
                self.assertEqual(d['graph_captures_during_run'],[0]*4,str(f))
                for r in d['outputs']:
                    self.assertEqual(r['ids'],expected[key][r['id']][:len(r['ids'])],str(f))
                checked+=1
        self.assertGreater(checked,0)


if __name__=='__main__':unittest.main()
