"""Preserve the physical probes and graph activity in the existing EPD archive format."""

from ..mllm_epd_prefill_heavy_modality_discovery import archive
from .online_evidence import response_text_digest


def main():
    archive.SCHEMA = 5
    archive.PROFILE_MEMBERS.add("graphs.csv")
    archive.STATE_MEMBERS.update({"burners_before.json", "burners_after.json"})
    capture_mode = archive.capture_mode
    client_fields = archive.client_fields

    def capture_idle_activity(relative):
        if relative.parts == ("raw", "vision_activity.jsonl"):
            return "jsonl"
        return capture_mode(relative)

    archive.capture_mode = capture_idle_activity

    def capture_response_digest(row):
        fields = client_fields(row)
        if row.get("status") == "MEASURED_ONLINE" and isinstance(value := row.get("text"), str):
            fields["normalized_response_text_SHA256"] = response_text_digest(value)
        return fields

    archive.client_fields = capture_response_digest
    archive.main()


if __name__ == "__main__":
    main()
