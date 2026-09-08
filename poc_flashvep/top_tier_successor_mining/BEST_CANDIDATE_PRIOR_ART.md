# SERE successor adversarial prior-art result

- [SERE](https://arxiv.org/abs/2602.07616) already studies calibration choices,
  similarity thresholds,primary-expert count and the prefill/decode distinction.
  Its AppendixC.2 explicitly says rerouting does not reduce FLOPs. The observed
  primary-set dilution is connected to an explicit algorithm contract.
- [XShare](https://arxiv.org/abs/2602.07265) is an adjacent expert-sharing adversary;
  generic sharing/similarity is not an independent novelty argument.
- [AnyExperts](https://arxiv.org/abs/2511.18314) already emphasizes semantic expert
  importance and sensitive OCR/NLP tasks. OCR fragility alone is insufficient.
- [ACE](https://arxiv.org/abs/2609.05228) provides parameter-derived contribution
  estimators. Its text-only setup is not proof of MLLM behavior,but a generic
  "better contribution score" successor would need an explicit distinction.

See PRIOR_ART_REFRESH_20260908.md for source-reading depth and caveats. None of
these papers' baseline numbers is imported as evidence of our implementation's
failure. No priority or first-observation claim is made. The immediate veto is
stronger than a speculative literature collision: existing S/rho tuning already
repairs most of the actual measured quality loss.
