# Reproduction commands

Run commands from `/home/esjung/MLLM-EP-github`.
Use `/home/esjung/.venvs/looped-pd-abc/bin/python` as `PYTHON` below.
Create a new run name before repeating an immutable completed configuration.
Do not delete an old namespace to force a rerun.

```bash
$PYTHON -m poc_flashvep.looped_pd_headroom_recovery.tests
$PYTHON -m poc_flashvep.looped_pd_headroom_recovery.run NEW_CONFIGURATION_NAME
$PYTHON -m poc_flashvep.looped_pd_headroom_recovery.analyze
$PYTHON -m poc_flashvep.looped_pd_headroom_recovery.capacity_analysis
$PYTHON -m poc_flashvep.looped_pd_headroom_recovery.queues
$PYTHON -m poc_flashvep.looped_pd_headroom_recovery.gates
$PYTHON -m poc_flashvep.looped_pd_headroom_recovery.accounting
$PYTHON -m poc_flashvep.looped_pd_headroom_recovery.paired_analysis
$PYTHON -m poc_flashvep.looped_pd_headroom_recovery.reporting
$PYTHON -m poc_flashvep.looped_pd_headroom_recovery.figures
$PYTHON -m poc_flashvep.looped_pd_headroom_recovery.timeline_figures
```

Each `configs/NAME.json` includes the complete input IDs and arrival offsets or names their pinned dataset.
Each `raw/NAME/source.zip` records the actual executed harness and private runtime diff.
`commands.jsonl` records the launch, config hash, source archive hash, and execution baseline commit.
`raw/NAME/isolation.json` records physical GPU IDs, UUIDs, driver, and other active processes.
`state/preservation.json` records the original Git index signature and private runtime diff signature.

Publication uses a separate results branch and temporary index.
It does not alter the active staged index.
Use `archive_evidence restore --destination EXPLICIT_EMPTY_DIRECTORY` to reconstruct oversized evidence from lossless archive parts.
