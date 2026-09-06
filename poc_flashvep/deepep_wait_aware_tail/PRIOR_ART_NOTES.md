# Prior-art notes

* DeepEP already implements asynchronous NVSHMEM/NVLink dispatch and combine
  with `previous_event` stream dependencies.  This PoC does not claim that
  synchronization itself is new.
* vLLM's `DeepEPHTPrepareAndFinalize` passes `previous_event` to layout,
  dispatch, and combine and waits on returned `EventOverlap` objects before
  expert/finalize use.  With DBO disabled, the vLLM `dbo_get_previous_event`
  helper returns `None`; internal DeepEP notification/barrier state remains.
* ASAP, Gimbal, and other MoE scheduling work study broader asynchronous or
  DP/EP scheduling issues.  The narrow candidate here is a runtime-visible
  outstanding communication-state signal used for selective draining, not
  global barrier removal or a new communication kernel.
* The present evidence is not sufficient for a novelty claim: the installed
  API does not expose native readiness, and selective stream drain was not
  robust across independent runs.  Any method work requires a literature
  re-check and a lower-overhead readiness API first.
