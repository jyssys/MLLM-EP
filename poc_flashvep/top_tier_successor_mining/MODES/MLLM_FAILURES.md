# MoDES failures versus calibration/source controls

## Final GPU resumption supersedes the historical checkpoint below

Final resumption: official1024-question/grid100 frontier completed,436 unique evaluations. Held-out256-question quality and corrected six-condition EP screen completed;three independent GQA-B16 engines do not reproduce the initial large positive point. Known OCR trade-off and port-cost floor do not establish a nontrivial successor. Kimi and new prototype not run.

## Historical audit / release record

ChartQA128 pilot: vanilla89.0625%; roughly85%-skip calibration point85.9375%
(-3.125pp,95%CI[-7.8125,.78125]); roughly90%-skip point77.34375%
(-11.71875pp,95%CI[-17.96875,-5.46875]). Aggressive OCR loss is already reported
by MoDES, and these policies predate the full official calibration.

New CPU check over462 observed fixed-other-threshold comparisons finds10 skip
fraction decreases when a threshold increases;5 exceed.1pp, none exceed1pp,
maximum.90345pp. Larger reversals occur at high text thresholds with very large
KL, far from the low-KL selected point. Joint downstream routing can change.
This does NOT establish a material calibration-search failure or an unseen better
grid optimum. Do not promote a mathematical caveat without Pareto headroom.

Obvious attacks remain larger/domain-specific calibration, safer thresholds and
fallback. Their retained serving speed has not been measured. No new MLLM-specific
failure has survived these attacks yet; Kimi transfer remains untested here.
