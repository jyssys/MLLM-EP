"""Use frozen non-ephemeral NIXL ports without changing compiler hashing."""
from pathlib import Path
import socket


def choose(cfg,role,excluded=()):
    key=','.join(map(str,cfg['gpus']));mapping=cfg['fixed_NIXL_control_ports']
    pair=mapping[key];assert len(pair)==2 and pair[0]!=pair[1]
    port=pair[('prefill','decode').index(role)]
    assert isinstance(port,int) and 1024<port<65536 and port not in excluded
    low,high=map(int,Path('/proc/sys/net/ipv4/ip_local_port_range').read_text().split())
    assert not low<=port<=high
    # Do not substitute a random port if this address is occupied.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        probe.bind(('127.0.0.1',port))
    return port
