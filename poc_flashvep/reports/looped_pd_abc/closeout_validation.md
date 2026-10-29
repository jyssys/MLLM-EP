# Closeout checks

The scientific review is complete with explicit limitations.
No new GPU inference starts during this closeout.
The measured runtime patch remains unchanged at SHA-256 `4904a311358c5cf6467678d0e1c30af38b5a96064cae4c96ed680943d7e694c7`.

## CPU checks

| Check | Result | Log |
|---|---|---|
| Report accounting, scope, and synchronous fixtures | 35 passed | `logs/final_closeout_cpu_tests_sync_fixtures.log` |
| Actual worker append seam, registered asynchronous protocol | 1 passed | `logs/final_closeout_cpu_tests_async_seam.log` |
| Earlier mixed-mode invocation without asynchronous hook | 35 passed, 1 failed | `logs/final_closeout_cpu_tests.log` |
| Earlier mixed-mode invocation with asynchronous hook everywhere | 34 passed, 2 failed | `logs/final_closeout_cpu_tests_registered_protocol.log` |

The first failure uses synchronous append in a test that requires asynchronous append.
The next two failures use synchronous file-read fixtures before asynchronous writes drain.
Separate recorder modes resolve those test-environment failures.
No inference, model, scheduler, or recorder implementation changes in this closeout.
The failing logs remain available. They do not become passing GPU measurements.

## Scientific gates

All 104 repeated comparison rows have three matched frozen starts.
The final strongest-E2E table has 18 rows. The complete strongest-baseline SLO grid has 378 rows.
The SLO grid preserves all declared thresholds. Favorable cells remain exploratory.
The final tables do not contain an accepted original-model-equivalent serving GO.
Official-reference sequence differences and C's batched-output differences remain explicit.

The complete raw-evidence checksum list is `raw_evidence_manifest.csv`.
It preserves original request outputs, source ZIPs, transfer checks, profiles, and failure status files.
The final report input hashes are in `final_review_input_hashes.json`.
The queue totals and two optional profile failures are in `final_manifest_status.json`.

## Git and resources

The release includes only `poc_flashvep/looped_pd_abc/` and `poc_flashvep/reports/looped_pd_abc/`.
The outside staged-index hash must remain `c162445edb25097abbf18627e41ba8dffc2a15f9a35e6d11385b37a9eb449a22`.
Original dirty and staged files outside those paths remain untouched.
Raw CSV CRLF endings and original log whitespace remain unchanged. They are evidence, not prose formatting defects.
Owned burners run on physical GPUs 0–7 after the measurement queue finishes.
The final response states the actual release SHA and the independently checked remote SHA.
