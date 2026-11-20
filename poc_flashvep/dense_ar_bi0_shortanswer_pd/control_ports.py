"""Keep NIXL listeners outside the OS ephemeral allocation range."""
import random
import socket
from pathlib import Path


def candidate_ports(ephemeral):
    low,high=ephemeral
    return [p for p in list(range(18000,28000))+list(range(62000,65000)) if not low<=p<=high]


def choose(excluded=()):
    ephemeral=tuple(map(int,Path('/proc/sys/net/ipv4/ip_local_port_range').read_text().split()))
    choices=candidate_ports(ephemeral);random.SystemRandom().shuffle(choices)
    for port in choices:
        if port in excluded:continue
        try:
            with socket.socket() as probe:probe.bind(('127.0.0.1',port))
        except OSError:continue
        return port
    raise RuntimeError('No available non-ephemeral NIXL control port')


def call_listener(original,errors,encoded,ready,*args):
    try:return original(encoded,ready,*args)
    except Exception as error:
        errors.append(repr(error));ready.set()
