# Reference versus EP quality

MMStar paired result: `{"baseline_correct_to_wrong": 29, "baseline_wrong_to_correct": 35, "cluster_bootstrap_95ci_pp": [-0.6, 1.4666666666666666], "ep_accuracy": 0.7146666666666667, "ep_minus_reference_pp": 0.4, "exact_output_agreement": 0.8073333333333333, "image_clusters": 1500, "noninferiority_margin_pp": -1.0, "noninferiority_proven": true, "paired_n": 1500, "reference_accuracy": 0.7106666666666667, "schema_version": "mllm_ep_paired_quality.v1", "task": "mmstar"}`

ChartQA paired result: `{"baseline_correct_to_wrong": 18, "baseline_wrong_to_correct": 13, "cluster_bootstrap_95ci_pp": [-0.5191075767616523, 0.28164347249834326], "ep_accuracy": 0.854, "ep_minus_reference_pp": -0.12701568367572344, "exact_output_agreement": 0.9564, "image_clusters": 1509, "noninferiority_margin_pp": -1.0, "noninferiority_proven": true, "paired_n": 2500, "reference_accuracy": 0.856, "schema_version": "mllm_ep_paired_quality.v1", "task": "chartqa"}`

Non-inferiority is claimed only when the cluster-bootstrap lower CI excludes
-1.0 percentage point. Lack of a significant difference is not equivalence.
