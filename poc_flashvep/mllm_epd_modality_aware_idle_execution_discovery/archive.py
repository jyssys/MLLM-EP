"""Preserve the physical probes and graph activity in the existing EPD archive format."""

from ..mllm_epd_prefill_heavy_modality_discovery import archive
from .online_evidence import response_text_digest


def main():
    archive.SCHEMA = 6
    archive.PROFILE_MEMBERS.add("graphs.csv")
    archive.STATE_MEMBERS.update({"burners_before.json", "burners_after.json"})
    capture_mode = archive.capture_mode
    client_fields = archive.client_fields
    sanitize_line = archive.sanitize_line

    def capture_idle_activity(relative):
        if len(relative.parts) == 2 and relative.parts[0] == "raw" and relative.name in {
            "vision_activity.jsonl", "ready_trials.jsonl", "ready_prime.jsonl",
            "ready_retired_decode.jsonl", "ready_retired_prefill.jsonl"
        }:
            return "jsonl"
        return capture_mode(relative)

    archive.capture_mode = capture_idle_activity

    def sanitize_ready_line(row, mode):
        if mode == "jsonl" and {"arm", "fixture", "error_type"} <= row.keys():
            row.pop("detail", None)
        return sanitize_line(row, mode)

    archive.sanitize_line = sanitize_ready_line

    def capture_response_digest(row):
        fields = client_fields(row)
        if row.get("status") == "MEASURED_ONLINE" and isinstance(value := row.get("text"), str):
            fields["normalized_response_text_SHA256"] = response_text_digest(value)
        return fields

    archive.client_fields = capture_response_digest
    archive.main()


if __name__ == "__main__":
    main()
