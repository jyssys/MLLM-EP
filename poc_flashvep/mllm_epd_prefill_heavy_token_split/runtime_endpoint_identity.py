"""Do not accept an unrelated listener as a newly initialized coordinator."""
import json
import socket


def require_identity(response,expected_name,process):
    assert response.status==200
    value=json.load(response)
    assert process.poll() is None,'The new coordinator terminated during readiness observation'
    assert value.get('name')==expected_name,('Wrong coordinator identity',expected_name,value.get('name'))
    return value


def require_free_ports(ports):
    for port in ports:
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            try:probe.bind(('127.0.0.1',port))
            except OSError as error:
                raise AssertionError(f'Port{port} is occupied. Inspect its actual owner before GPU initialization.') from error
