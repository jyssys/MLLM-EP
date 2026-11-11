# Ten-hour campaign review

Review time: 2026-11-10 23:26:08 UTC.
Start time: 2026-11-10 13:25:48 UTC.
The review occurs 10 hours and 20 seconds after the start.
It is not a final verdict.

## Evidence at the review

The GPU campaign has a reproducible exact fixed-R32 Native fast path.
The all-checkpoint no-migration control still has a large TTFT regression after wire matching.
Selective checkpoint exposure preserves much of the fast path.
On the primary selection trace, the adaptive Loop arm exposes three requests and moves none.
The matched previous Loop policy moves 10 Loops on three requests and has worse TTFT than Zero.

An exact forced Loop31 can reduce the P queue and TTFT.
The Decode-priority intervention completes 128 actual continuations.
Its TTFT p95 is 7238.13 milliseconds.
Its ITL p99 is 104.68 milliseconds.
The stronger full-device Zero measures ITL p99 of 88.48 milliseconds.
The intervention fails the relative Decode gate.

The 128-token Loop tiling increases observed D submissions from 128 to 2048.
It is slower than the untiled arm on both TTFT p95 and ITL p99.
The initial Token quantum request does not constrain its staged Prelude.
The measured Token arm submits 32 Loops per request.
The corrected implementation must pass actual geometry and state checks.

The 2048/4 discovery reduces observed D load and retains P backlog.
Its Zero N128 completes.
The adaptive Loop N8 qualification moves four Loops on three requests.
This controlled N8 result is not clean serving evidence.
The separate N32 qualification and N128 comparison remain in progress.

## Remaining work

The active serial owner has completed 24 of its 44 jobs.
Its current job is the 2048/4 Loop N32 qualification.
The next serial block waits for this owner to restore resources.
That block contains 15 cells.
It checks a fresh full-device Zero, corrected Token suffixes, and aligned Loop tiles.
No new GPU owner overlaps the active pair.

Reserve the held-out primary comparison before optional additional search.
Use at least three paired Arrival seeds and an independent repeated Zero.
Run short-output confirmation only if its clean selection warrants it.
Use remaining time for disjoint resource, fixed-dose, or early-transfer controls only when they answer a remaining question.
Do not begin another full kernel collector sweep.
Keep a separate verified stock AR diagnostic if the execution environment permits it.

Do not start a long cell after 2026-11-11 03:25:48 UTC.
Release the science GPU lane by 03:55:48 UTC.
Reserve the remaining time for evidence verification, publication, and restoration.
The operational safety stop remains 05:25:48 UTC.
