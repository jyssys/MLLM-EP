"""Auto-install the read-only resource-atlas NVTX hook in vLLM workers."""
import os

if os.environ.get("FLASHVEP_ATLAS_DISABLE") != "1":
    try:
        from poc_flashvep.resource_atlas.atlas_hook import install
        install()
    except Exception as exc:  # pragma: no cover - startup diagnostics
        path = os.environ.get("FLASHVEP_ATLAS_RESULT_DIR")
        if path:
            with open(os.path.join(path, f"hook_install_error_{os.getpid()}.txt"), "w") as f:
                f.write(repr(exc) + "\n")
