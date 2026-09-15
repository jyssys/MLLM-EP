"""Execute one HumanEval sample inside an unprivileged, networkless chroot.

The parent passes only a JSON sample through stdin. This program must never be
run directly on the host with untrusted model-generated code.
"""

import json
import resource
import signal
import sys


def main():
    payload = json.load(sys.stdin)
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    signal.alarm(3)
    result = {"passed": False, "failure": None}
    try:
        entry = payload["entry_point"]
        if not entry.isidentifier():
            raise ValueError("invalid official HumanEval entry point")
        code = payload["candidate"] + "\n" + payload["test"] + "\ncheck(" + entry + ")"
        compiled = compile(code, "<human-eval-sample>", "exec")
        scope = {}
        exec(compiled, scope, scope)
        result["passed"] = True
    except SyntaxError as exc:
        result["failure"] = "SYNTAX_ERROR"
        result["detail"] = str(exc)[:300]
    except BaseException as exc:
        result["failure"] = type(exc).__name__
        result["detail"] = str(exc)[:300]
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
