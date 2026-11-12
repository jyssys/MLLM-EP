import threading,unittest
from .async_ingress import AsyncIngress


class AsyncIngressTests(unittest.TestCase):
    def test_a_blocked_source_does_not_block_destination_dispatch(self):
        blocked=threading.Event();release=threading.Event();delivered=threading.Event();rows=[]
        class Source:
            def send(self,message):blocked.set();release.wait(1)
        class Destination:
            def send(self,message):delivered.set()
        sender=AsyncIngress(dict(prefill=Source(),decode=Destination()),lambda role,row:rows.append((role,row)))
        message=dict(kind='add',request_id='q0')
        sender.submit('prefill',message);self.assertTrue(blocked.wait(.5))
        sender.submit('decode',message);self.assertTrue(delivered.wait(.5))
        release.set();self.assertTrue(sender.close());sender.check()
        self.assertEqual(len(rows),2)

    def test_failed_rpc_is_not_silently_accepted(self):
        class Failed:
            def send(self,message):raise BrokenPipeError('test')
        sender=AsyncIngress(dict(prefill=Failed()),lambda *a:None)
        sender.submit('prefill',dict(kind='add',request_id='q0'));self.assertTrue(sender.close())
        with self.assertRaises(AssertionError):sender.check()


if __name__=='__main__':unittest.main()
