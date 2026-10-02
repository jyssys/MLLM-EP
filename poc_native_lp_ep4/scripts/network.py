"""Bounded external-DP rendezvous allocation, outside OS ephemeral ports."""
import random
import socket
from pathlib import Path


def choose_dp_port(block=16, attempts=64):
    low,high=map(int,Path('/proc/sys/net/ipv4/ip_local_port_range').read_text().split())
    candidates=[base for base in range(12000,30000-block)
                if base+block-1<low or base>high]
    if not candidates:
        candidates=[base for base in range(61000,65000-block)
                    if base+block-1<low or base>high]
    if not candidates:raise RuntimeError('no configured non-ephemeral DP port block')
    rng=random.SystemRandom()
    for _ in range(attempts):
        base=rng.choice(candidates);sockets=[]
        try:
            for port in range(base,base+block):
                sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
                sockets.append(sock);sock.bind(('127.0.0.1',port))
            return base
        except OSError:
            pass
        finally:
            for sock in sockets:sock.close()
    raise RuntimeError('no free consecutive external-DP rendezvous ports after bounded checks')
