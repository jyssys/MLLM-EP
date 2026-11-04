# Low-load execution order at 23:17 KST

Both models and both split axes passed the unforced N32 gate.
All four heavy-load N128 runs completed.
Huginn medium Token currently runs.

The remaining low-load Huginn arrival windows are longer than the Ouro windows.
The fixed new-job cutoff remains 23:35:34 KST.
Prioritize the two shorter Ouro low-load cells before the final Huginn low-load cells.
This ordering increases the chance that every cell starts before the cutoff.
It does not extend the GPU stop or final deadline.

The prioritized cells use the same immutable names and configuration builder as the main controller.
The dispatcher uses the existing namespace lock and global quiet lane.
The main controller later reuses the completed receipts.
Only one GPU pair runs at a time.
No input, arrival offset, model, graph policy, SLO, or online decision rule changes.
This is a budget-order change, not an extra tuning trial or a selected favorable result.

At 23:31 KST, queue the final qualified Huginn low-load Loop cell on the same quiet lane.
It can start only after the current GPU worker finishes and releases that lane.
The dispatcher still rechecks the original cutoff before launching the worker.
This avoids CPU postprocessing between the final two independent GPU cells.
It does not bypass the N32 gate or permit overlapping GPU workers.
