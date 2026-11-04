# Qualified projection geometry and split fairness

Evidence class: source audit and structural accounting. No GPU speedup is reported here.

The retained numerical patch uses fixed 256-row tiles for selected projections.
Ouro retains `down_norm`. Huginn retains `down_norm_adapter`.
The selected Huginn projections include MLP down projections and the recurrent adapter.
This geometry applies to short prompts and Decode rows too.
It is not the earlier `canonical_all` diagnostic that used a different small-row rule.

Source: `poc_flashvep/looped_pd_b_numerical_diagnosis/patches.py`.
The current campaign does not change this qualified numerical patch.

## Structural comparison

The table assumes one full suffix or prompt execution at each depth.
It counts rows for one selected fixed-tile projection per physical layer.
It does not count all model FLOPs or measured kernel work.

| Huginn L2048 split | D logical recurrent rows | D fixed-projection rows |
|---|---:|---:|
| Loop cut31, one remaining loop | 2048 | 2048 |
| Token prefix1984, suffix64 across R32 | 2048 | 8192 |
| Loop cut28, four remaining loops | 8192 | 8192 |
| Token prefix1792, suffix256 across R32 | 8192 | 8192 |

The initial cut31 comparison is a conservative interference test.
It is not sufficient evidence of intrinsic Loop-specific efficiency.
A measured advantage there could include the qualified patch's padding cost.
The aligned cut28 versus suffix256 control removes this particular row-count difference.
It does not remove attention, launch, transfer, or scheduling differences.

Both methods must keep the numerical patch and receive the same execution budget.
The Token method must not receive an artificially smaller chunk solely to create a speed difference.
If time or correctness prevents the aligned control, report it as NOT_MEASURED.
Do not attribute the unseparated padding effect to recurrent architecture.
