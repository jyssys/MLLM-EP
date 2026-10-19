"""Reproduce a proc-environment lease failure after the native title primitive."""
import argparse
from multiprocessing import Pipe
import os
import subprocess
import sys
from poc_ouro_parallelism_deep_discovery.contract import ROOT, write
from poc_ouro_parallelism_deep_discovery.dp4_cohort import collect_messages
from poc_ouro_parallelism_deep_discovery.tp2_dp2_cohort import inspect_process_leases


def probe(preserve_environment):
    parent, child = Pipe()
    identity = "CPU_TITLE_GUARD_FIXTURE"
    code = (
        "import os,sys; from multiprocessing.connection import Connection; "
        "from poc_ouro_parallelism_deep_discovery.runtime import configure_native_imports; "
        "configure_native_imports(); import setproctitle; "
        "setproctitle.setproctitle('OWNED_CPU_TITLE_PROBE'); "
        "c=Connection(int(sys.argv[1])); "
        "c.send(dict(kind='LOADED',worker=0,pid=os.getpid(),"
        "logical_guard_present=os.environ.get('LOOPED_KV_OUTPUT')=='CPU_TITLE_GUARD_FIXTURE',"
        "setproctitle_file=setproctitle.__file__)); "
        "assert c.recv()==dict(kind='STOP'); c.close()"
    )
    environment = dict(os.environ, LOOPED_KV_OUTPUT=identity)
    environment.pop("SPT_NOENV", None)
    if preserve_environment:
        environment["SPT_NOENV"] = "1"
    process = subprocess.Popen([sys.executable, "-c", code, str(child.fileno())],
                               env=environment, pass_fds=(child.fileno(),))
    child.close()
    try:
        row = collect_messages({0: parent}, "LOADED", 10)[0]
        try:
            leases = inspect_process_leases([process.pid], os.getpgid(0), os.getsid(0),
                                           identity, ancestor_pid=process.pid)
            row.update(proc_lease_pass=True, leases=leases)
        except RuntimeError as exc:
            row.update(proc_lease_pass=False, error=str(exc))
        parent.send(dict(kind="STOP"))
        process.wait(timeout=10)
        assert process.returncode == 0
        return dict(row, SPT_NOENV=preserve_environment, CUDA_context_created=False)
    finally:
        parent.close()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    rows = [probe(False), probe(True)]
    write(output, dict(rows=rows,
        hypothesis="Process-title rewriting can obscure the initial proc environment while Python retains the guard variable.",
        scope="Actual proc lease helper and installed title primitive. CPU children only. Not a native-engine rerun."))
    assert all(row["logical_guard_present"] for row in rows)
    assert not rows[0]["proc_lease_pass"]
    assert rows[1]["proc_lease_pass"]
    print("CPU_TITLE_REPRO: default lease FAIL, SPT_NOENV lease PASS, logical environment intact in both")


if __name__ == "__main__":
    main()
