# TEAM adaptation

Not run. TEAM adaptation is explicitly secondary and opens only after an
independent vanilla candidate passes the oracle/live gates. The best exact
vanilla oracle was 0.153% E2E and the primary rank-local method was 0%; therefore
`Vanilla / TEAM / Ours / TEAM+Ours` would not be a meaningful experiment.

TEAM is evaluated only after an independent vanilla candidate survives the
oracle and live gates. Otherwise this experiment is intentionally not run: a
negative vanilla direction must not be rescued by becoming a TEAM-specific
patch.
