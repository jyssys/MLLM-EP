# Stronger finite Prefill batching control

Declared before the new B4 measurements.

The accepted Ouro screens use two P compute slots.
A larger legal batch could improve the P service reference.
The existing reference must not be treated as a hardware maximum.

Test B1, B2, and B4 with three finite local repetitions for each model.
Use Ouro L4096 and Huginn L2048, with one generated token.
Keep weights, precision, recurrent depth, numerical fixes, and KV pool size unchanged.
Set four local slots and a token/graph row limit of four prompt lengths.
Compare each barrier execution with the native local output reference.
The reference separates Prefill from the first Coda token.

Retain OOM, unsupported geometry, and exactness failures.
Do not reduce the model or remove validity checks to fit the larger batch.
This is a P-only capacity control, not a same-run serving result.

If the stronger finite reference changes the 85% service-proximity outcome, report that change.
Retain the original ratios and their reference provenance.
Do not preserve an earlier PASS by ignoring a faster valid reference.
No new threshold is introduced.

The controller is `poc_flashvep.looped_pd_autonomous_headroom.batch_capacity_control`.
Its state ledger is `state/batch_capacity_control_results.json`.
