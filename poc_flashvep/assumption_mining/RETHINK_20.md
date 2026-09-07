# Rethink after 20 assumptions

**Expected:** host metadata and DP coordination are structural taxes.
**Observed:** metadata is ~0.11 ms typical and DP controls are null.
**Failed assumption:** frequency implies economic significance.
**New system fact:** repeated tiny work can be less important than rare but
critical work, yet the rare work must still be request-visible.

New children: (1) persistent device metadata lifecycle, (2) scoped DP
rendezvous, (3) workspace/allocator state contract.  Only the last remains
UNKNOWN; the first two are below the analytical gate.
