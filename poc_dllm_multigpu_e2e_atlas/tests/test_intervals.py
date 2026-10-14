import unittest
from poc_dllm_multigpu_e2e_atlas.intervals import union_ms,exclusive_budget,kernel_partition


class IntervalTests(unittest.TestCase):
    def test_overlap_not_double_counted(self):
        self.assertEqual(union_ms([(0,5),(3,8),(9,10)]),9)
        parts=kernel_partition([dict(start=0,end=5,category='compute'),dict(start=3,end=8,category='NCCL')],0,10)
        self.assertEqual(sum(p['end']-p['start'] for p in parts),10)
        self.assertEqual(sum(p['end']-p['start'] for p in parts if p['category']=='OVERLAP_MULTIPLE_CATEGORIES'),2)

    def test_nested_wall_accounting(self):
        spans=[dict(id=0,parent=None,kind='root',host_start_ns=0,host_end_ns=10),
            dict(id=1,parent=0,kind='child',host_start_ns=2,host_end_ns=6)]
        parts=exclusive_budget(spans,0,10)
        self.assertEqual(sum(p['end']-p['start'] for p in parts),10)
        self.assertEqual(sum(p['end']-p['start'] for p in parts if p['kind']=='child'),4)


if __name__=='__main__':unittest.main()
