import json
from pathlib import Path
import tempfile
import unittest
from .publication import restricted_probe


class PublicationTests(unittest.TestCase):
    def test_partial_producer_debug_tensor_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            (folder/'configuration.json').write_text(json.dumps(dict(axis_kind='Layer',numerical_probe_sync=True)))
            self.assertTrue(restricted_probe(folder/'numerical_prefill_q0.pt'))
            self.assertFalse(restricted_probe(folder/'numerical_decode_q0.pt'))
            self.assertFalse(restricted_probe(folder/'axis_events_prefill.json'))
            (folder/'configuration.json').write_text(json.dumps(dict(axis_kind='Token',numerical_probe_sync=True)))
            self.assertFalse(restricted_probe(folder/'numerical_prefill_q0.pt'))

    def test_unsynchronized_debug_values_remain_local(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            (folder/'configuration.json').write_text(json.dumps(dict(numerical_probe=True)))
            self.assertTrue(restricted_probe(folder/'numerical_single_q0.pt'))
            self.assertFalse(restricted_probe(folder/'prompt_state_single.json'))


if __name__=='__main__':unittest.main()
