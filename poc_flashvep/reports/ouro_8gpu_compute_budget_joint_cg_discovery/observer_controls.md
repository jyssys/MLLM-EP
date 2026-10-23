# Ready-stream observer qualification

Two quiet GPU/model-process blocks compare clean and event lanes on all six native paths.
Every lane uses the same completion fairness, C512, requests, and generation cap.
Lane order reverses across source cohorts.
The output token IDs matched in 143/144 request pairs.
The single mismatch occurred in native Mixed at P2033/N4, trial 5.
Its cause remains unresolved. The observer can change batch geometry through host timing.
This possibility does not prove a numerical cause for this specific mismatch.
Individual cohort event/clean wall ratios ranged from 0.972051 to 1.043542.
The range is not a precise causal overhead estimate. Only two process blocks were measured.
Event service data remain diagnostic. Clean client timing is a separate lane.
No benchmark accuracy or capacity claim follows from this control.
