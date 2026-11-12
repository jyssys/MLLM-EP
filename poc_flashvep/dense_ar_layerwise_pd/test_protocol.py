import unittest
from .protocol import Frontier,Phase


class ProtocolTests(unittest.TestCase):
    def test_each_cut(self):
        for cut in (28,32,34,35):
            f=Frontier('q0',1024,cut);f.export()
            self.assertFalse(f.decode_allowed())
            f.import_state(kv_complete=True,hidden_complete=True)
            f.complete(cut,36);self.assertTrue(f.decode_allowed())
            f.release(native_ack=True);self.assertEqual(f.phase,Phase.RELEASED)

    def test_missing_or_duplicate_work_rejected(self):
        f=Frontier('q0',1024,32);f.export();f.import_state(kv_complete=True,hidden_complete=True)
        with self.assertRaises(AssertionError):f.complete(31,36)
        with self.assertRaises(AssertionError):f.complete(33,36)
        f.complete(32,36)
        with self.assertRaises(AssertionError):f.complete(32,36)

    def test_import_and_release_require_completion(self):
        f=Frontier('q0',1024,32);f.export()
        with self.assertRaises(AssertionError):f.import_state(kv_complete=False,hidden_complete=True)
        with self.assertRaises(AssertionError):f.release(native_ack=False)
        with self.assertRaises(AssertionError):f.complete(32,36)

    def test_source_ACK_can_precede_destination_compute(self):
        f=Frontier('q0',8000,32);f.export();f.import_state(kv_complete=True,hidden_complete=True)
        f.release(native_ack=True)
        self.assertEqual(f.phase,Phase.IMPORTED);self.assertFalse(f.decode_allowed())
        with self.assertRaises(AssertionError):f.release(native_ack=True)
        f.complete(32,36);self.assertEqual(f.phase,Phase.RELEASED)
        self.assertTrue(f.decode_allowed())


if __name__=='__main__':unittest.main()
