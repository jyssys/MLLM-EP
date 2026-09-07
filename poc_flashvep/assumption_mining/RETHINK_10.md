# Rethink after 10 assumptions

**Expected:** dependency relaxation would dominate the search.
**Observed:** the only large event is a communication wait with 1.09% direct
request mass.
**Failed assumption:** a spectacular rank-local tail is automatically a large
request-level opportunity.
**New system fact:** critical-path accounting must precede method design.

New non-cosmetic children: (1) token-scoped readiness, (2) engine/worker
ownership boundary, (3) cross-request slack borrowing.  All are retained only
if a direct request-level oracle can be shown.
