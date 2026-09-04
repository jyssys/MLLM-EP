"""Install the crossover worker hook when explicitly enabled."""
import os

if os.environ.get("FLASHVEP_CROSSOVER_ENABLE") == "1":
    try:
        from poc_flashvep.modality_aware_tp_ep_crossover.worker_hook import install

        install()
    except Exception as exc:  # pragma: no cover
        path = os.environ.get("FLASHVEP_CROSSOVER_RAW_DIR")
        if path:
            with open(os.path.join(path, f"hook_error_{os.getpid()}.txt"), "w") as f:
                f.write(repr(exc) + "\n")
