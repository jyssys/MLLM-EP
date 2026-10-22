"""CPU-only data completion. Reports still require scientific review before Git."""

import json
import os

from .common import ROOT, write


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    from . import (
        analyse,
        controls,
        nsys_analysis,
        figures,
        report_tables,
        inventory,
        verify,
    )

    analyse.main()
    controls.main()
    nsys_analysis.main()
    figures.main()
    report_tables.main()
    inventory.main()
    verify.main()
    state = json.loads((ROOT / "research_state.json").read_text())
    state.update(
        status="DATA_CAPTURE_AND_CPU_ANALYSIS_COMPLETE_REPORT_REVIEW_PENDING",
        next="Review final CSVs and figures. Update provisional Markdown. Verify timing and resources. Archive evidence. Commit/push only this study.",
        counts=json.loads((ROOT / "analysis_counts.json").read_text()),
        burner_restore="Resource wrapper completed its restore call before CPU analysis.",
        pending_commands=[],
    )
    write(ROOT / "research_state.json", state)
    print("DATA_READY_FOR_REPORT_REVIEW")


if __name__ == "__main__":
    main()
