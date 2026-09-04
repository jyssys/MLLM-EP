"""Install hooks only inside vLLM workers, never in the outer DP driver."""

import os

try:
    from poc_flashvep.deepep_revalidation.backend_probe import install_backend_probe
    install_backend_probe()
except Exception:
    pass

# The outer DP driver starts before it sets its per-run environment.  Do not
# import/patch vLLM there: importing the distributed/worker stack before the
# engine is initialized can leave a driver waiting on the v1 shared-memory
# broadcast.  Child workers inherit FLASHVEP_MATRIX_ENABLE after the driver
# has configured it, so instrumentation is installed only in those workers.
if os.environ.get("FLASHVEP_DEEPEP_PROOF_DIR"):
    try:
        from poc_flashvep.live_traffic_matrix_validation.instrumentation import install
        install()
    except Exception as exc:
        try:
            from pathlib import Path
            p = os.environ.get("FLASHVEP_MATRIX_RAW_DIR", ".")
            Path(p).mkdir(parents=True, exist_ok=True)
            (Path(p) / f"hook_timing_error_pid{os.getpid()}.txt").write_text(
                f"{type(exc).__name__}: {exc}\n")
        except Exception:
            pass

    try:
        from poc_flashvep.mllm_ep8_critical_rank_coalescing_poc.route_capture import install as install_routes
        install_routes()
    except Exception as exc:
        try:
            from pathlib import Path
            p = os.environ.get("FLASHVEP_ROUTE_RAW_DIR", ".")
            Path(p).mkdir(parents=True, exist_ok=True)
            (Path(p) / f"hook_route_error_pid{os.getpid()}.txt").write_text(
                f"{type(exc).__name__}: {exc}\n")
        except Exception:
            pass
