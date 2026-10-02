import socket
import unittest
from pathlib import Path
from unittest.mock import patch

from poc_native_lp_ep4 import light_trace
from poc_native_lp_ep4.scripts.network import choose_dp_port


class RepairTest(unittest.TestCase):
    def test_port_block_is_free_and_non_ephemeral(self):
        base=choose_dp_port(block=4)
        low,high=map(int,Path('/proc/sys/net/ipv4/ip_local_port_range').read_text().split())
        self.assertTrue(base+3<low or base>high)
        sockets=[]
        try:
            for port in range(base,base+4):
                sock=socket.socket();sockets.append(sock);sock.bind(('127.0.0.1',port))
        finally:
            for sock in sockets:sock.close()

    def test_events_reused_only_across_drained_invocations(self):
        class Event:
            def __init__(self,**kwargs):self.records=0
            def record(self):self.records+=1
        light_trace.EVENT_POOL.clear();light_trace.CALL_SLOTS.clear()
        kernel=object()
        with patch.object(light_trace.torch.cuda,'Event',Event):
            first=light_trace.pooled_events(kernel,0)
            second=light_trace.pooled_events(kernel,0)
            self.assertIsNot(first,second)
            light_trace.CALL_SLOTS.clear()
            again=light_trace.pooled_events(kernel,0)
            self.assertIs(again,first)
            self.assertTrue(all(e.records==1 for e in again))
        light_trace.EVENT_POOL.clear();light_trace.CALL_SLOTS.clear()


if __name__=='__main__':unittest.main()
