# Blind EP2 to EP4 transfer

The EP2-only frozen predictor was applied to actual EP4 untouched requests.
This is a transfer diagnostic, not EP8 validation.

`{"request_ranking_and_gain": {"gain_error_median_pp": 18.36970981945918, "gain_error_p90_pp": 87.47395837386239, "material_pairs": 22, "materiality_rule": "actual symmetric difference >=5%", "scenario_pairs": 28, "sign_accuracy": 0.8636363636363636, "spearman": 0.880952380952381}, "request_routed_stage": {"median_absolute_error_ms": 6.547199505911596, "median_ape_percent": 11.417852068401821, "median_signed_bias_ms": 3.857060117508656, "n": 8, "p90_ape_percent": 17.688048107979945}, "status": "MEASURED"}`

The blind transfer median/P90 APE is
11.418% /
17.688%.
Its residual distribution defines EP8 optimistic/base/stressed communication
sensitivity and forbids a calibrated EP8 point-latency claim.
