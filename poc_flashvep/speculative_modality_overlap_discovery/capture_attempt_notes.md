# Capture attempt notes

1. Conda `flashvep-poc` interpreter: vLLM initialized but the installed
   DeepEP extension was absent (`has_deep_ep() == False`).  This attempt was
   stopped before request capture and is not used as evidence.
2. Validated `/home/esjung/.venvs/flashvep-deepep-v020` interpreter without the
   hook: real `DeepEPHTAll2AllManager` and `DeepEPHTPrepareAndFinalize` were
   active, but no diagnostic raw tensors were emitted because the opt-in
   `sitecustomize` hook was not installed early enough.
3. Final run: the existing hook and backend proof were installed explicitly in
   each worker before importing vLLM.  Six real images completed and produced
   exact top-8 reconstruction checks plus partial-output measurements.

These are runtime/harness corrections only; no model or routing semantics were
changed.
