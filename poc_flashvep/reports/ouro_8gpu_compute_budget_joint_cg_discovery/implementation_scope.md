# Implementation scope and current status

The external runtime is a new worktree at the pinned RLT commit.
It retains the previously qualified G2/G3 patch.
Historical checkouts and reports remain unchanged.

New opt-in Python adapters belong to `poc_flashvep/ouro_budget_joint_cg/`.
`fairness.py` credits native coda host delivery, not an issued recurrent batch.
The debt clears after at least one eligible request emits a token.
It does not require all active requests to complete together.
Cancellation and request-object identity prevent stale debt after ID reuse.
A 32-decode-turn escape prevents indefinite prefill starvation.
The escape is not recorded as token-completion credit.

`actions.py` selects new C per request, not through global config mutation.
An existing staged span remains immutable.
Continuation G clips to the remaining R4 depth.
The native scheduler/executor guard still defers incompatible G3/G1 batch rows.
Native request-event waits and KV ownership remain intact.

`service_observer.py` starts its event after native request wait dependencies.
It includes GPU preparation and the native model execution on that stream.
Queueing, CPU dispatch, event alignment, and client clocks remain separate fields.
The interval can include CPU launch gaps and is not pure kernel-busy time.
The observer never reads unfinished GPU tensors.

Sixteen CPU tests passed before pretrained GPU qualification.
They cover fairness, legal frontiers, enumeration, budget feedback, and budget admission.
Integration tests exercise the actual native scheduler, executor, and cache state machine with tiny software fixtures.
Those fixtures are not pretrained-model quality evidence.
Pretrained numerical probes and new cost collection are registered but not yet complete.

The new optimizer enumerates one legal action at a time.
Its completion-aware utility includes every remaining R4 group and staged prelude.
Budget admission defers unaffordable prefills while decode can progress.
Idle progress and a 250-ms starvation escape can exceed the budget.
The trace labels those exceptions explicitly.
All adaptive controls admit at most one prefill span per batch.
This prevents an unmodeled packed batch from exceeding the single-span budget silently.
Native fixed and Mixed controls retain native packing and remain separate baselines.
The controller and path lookup code exist, but no new adaptive serving result exists yet.
The previous completed-token D1 predictor is not used.
Native-vLLM Loop PP remains paused.
