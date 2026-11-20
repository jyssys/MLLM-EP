"""Keep archived accounting distinct from queue-causality claims."""
import unittest
from .historical_bridge_analysis import overlap,external_id
from .destination_chain import decompose


class HistoricalBoundaryTest(unittest.TestCase):
    def test_disjoint_intervals(self):
        self.assertEqual(overlap(1,2,3,4),0)

    def test_contained_interval(self):
        self.assertEqual(overlap(1,5,2,4),2)

    def test_touching_intervals(self):
        self.assertEqual(overlap(1,2,2,3),0)

    def test_agent_metadata_has_no_request(self):
        self.assertIsNone(external_id(None))

    def test_external_request_id(self):
        self.assertEqual(external_id('q12-workerUUID'),'q12')

    def test_invalid_nonstring_notification(self):
        self.assertIsNone(external_id(0))

    def test_exact_nonstarved_destination_chain(self):
        rows=[dict(arrival=i,ready=2+2*i,D_submit=2+3*i,first_worker=3+3*i,
            last_worker=4+3*i,finish=5+3*i) for i in range(3)]
        result=decompose(rows)
        self.assertAlmostEqual(sum(result['exact_parts_ms'].values()),5000)
        self.assertEqual(result['ready_supply_missing_fraction'],0)

    def test_exact_source_starved_destination_chain(self):
        rows=[dict(arrival=i,ready=t,D_submit=t,first_worker=t+1,last_worker=t+2,finish=t+3)
            for i,t in enumerate((2,10,20))]
        result=decompose(rows)
        self.assertAlmostEqual(sum(result['exact_parts_ms'].values()),32000/3)
        self.assertEqual(result['ready_supply_missing_fraction'],1)


if __name__=='__main__':unittest.main()
