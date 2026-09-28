
# Frozen held-out signature predictor comparison

Discovery rows are the only predictor memory; held-out image hashes are never added. All feature families use the same 3-NN predictor and normalization.

| window | predictor | expert hist cosine | top8 recall | rank cosine | rank-load MAE |
|---|---|---:|---:|---:|---:|
| 1-32 | P_text | 0.9676 | 0.9760 | 0.9978 | 0.0185 |
| 1-32 | P_vision | 0.9668 | 0.9761 | 0.9978 | 0.0185 |
| 1-32 | P_full | 0.9669 | 0.9763 | 0.9978 | 0.0185 |
| 1-32 | P_separated | 0.9670 | 0.9758 | 0.9978 | 0.0185 |
| 33-64 | P_text | 0.9161 | 0.9160 | 0.9942 | 0.0303 |
| 33-64 | P_vision | 0.9181 | 0.9170 | 0.9944 | 0.0299 |
| 33-64 | P_full | 0.9165 | 0.9171 | 0.9942 | 0.0302 |
| 33-64 | P_separated | 0.9178 | 0.9169 | 0.9943 | 0.0301 |
| 65-96 | P_text | 0.9148 | 0.9079 | 0.9940 | 0.0310 |
| 65-96 | P_vision | 0.9144 | 0.9079 | 0.9939 | 0.0313 |
| 65-96 | P_full | 0.9143 | 0.9067 | 0.9939 | 0.0313 |
| 65-96 | P_separated | 0.9142 | 0.9082 | 0.9939 | 0.0313 |
| 97-128 | P_text | 0.9143 | 0.8998 | 0.9941 | 0.0308 |
| 97-128 | P_vision | 0.9119 | 0.8986 | 0.9939 | 0.0312 |
| 97-128 | P_full | 0.9125 | 0.8990 | 0.9939 | 0.0312 |
| 97-128 | P_separated | 0.9129 | 0.8993 | 0.9940 | 0.0310 |
| 129-192 | P_text | 0.9113 | 0.9135 | 0.9938 | 0.0314 |
| 129-192 | P_vision | 0.9104 | 0.9132 | 0.9938 | 0.0314 |
| 129-192 | P_full | 0.9108 | 0.9130 | 0.9939 | 0.0312 |
| 129-192 | P_separated | 0.9116 | 0.9131 | 0.9939 | 0.0312 |
| 193-256 | P_text | 0.8843 | 0.8632 | 0.9916 | 0.0365 |
| 193-256 | P_vision | 0.8854 | 0.8646 | 0.9918 | 0.0362 |
| 193-256 | P_full | 0.8853 | 0.8648 | 0.9918 | 0.0362 |
| 193-256 | P_separated | 0.8846 | 0.8651 | 0.9918 | 0.0362 |
| 1-256 | P_text | 0.9820 | 0.9976 | 0.9988 | 0.0134 |
| 1-256 | P_vision | 0.9814 | 0.9976 | 0.9988 | 0.0136 |
| 1-256 | P_full | 0.9812 | 0.9976 | 0.9988 | 0.0137 |
| 1-256 | P_separated | 0.9815 | 0.9975 | 0.9988 | 0.0136 |

Aggregate paired request-bootstrap deltas:

- P_vision vs P_text: `{"active_expert_set_recall": {"beneficial_delta": 0.0, "ci95_high": 0.0, "ci95_low": 0.0, "requests": 80}, "decode_expert_top8_recall": {"beneficial_delta": 0.0, "ci95_high": 0.00013020833333333152, "ci95_low": -0.00013020833333333425, "requests": 80}, "expert_histogram_cosine": {"beneficial_delta": -0.0006360182650042318, "ci95_high": -2.202956795258466e-05, "ci95_low": -0.0012945148359006292, "requests": 80}, "expert_histogram_jsd": {"beneficial_delta": -0.0005046081566778023, "ci95_high": 0.0005545267515553063, "ci95_low": -0.0015573531208832796, "requests": 80}, "hottest_rank_accuracy": {"beneficial_delta": 0.0, "ci95_high": 0.0, "ci95_low": 0.0, "requests": 80}, "per_rank_load_mae": {"beneficial_delta": -0.00021665949914933727, "ci95_high": -1.0359919744233255e-06, "ci95_low": -0.00046535579821810845, "requests": 80}, "rank_distribution_cosine": {"beneficial_delta": -5.159743551377549e-05, "ci95_high": -1.413161154176032e-05, "ci95_low": -9.408628421681747e-05, "requests": 80}, "rank_distribution_jsd": {"beneficial_delta": -6.343293207766224e-05, "ci95_high": -1.3199594126763358e-05, "ci95_low": -0.0001202290250035729, "requests": 80}}`
- P_separated vs P_text: `{"active_expert_set_recall": {"beneficial_delta": 0.0, "ci95_high": 0.0, "ci95_low": 0.0, "requests": 80}, "decode_expert_top8_recall": {"beneficial_delta": -6.510416666666574e-05, "ci95_high": 6.51041666666699e-05, "ci95_low": -0.00022786458333333152, "requests": 80}, "expert_histogram_cosine": {"beneficial_delta": -0.0005130694418466572, "ci95_high": 0.00011434604604777221, "ci95_low": -0.0012223523364909194, "requests": 80}, "expert_histogram_jsd": {"beneficial_delta": -0.0002812292552439795, "ci95_high": 0.0006872447694418448, "ci95_low": -0.0012854532640228105, "requests": 80}, "hottest_rank_accuracy": {"beneficial_delta": 0.0, "ci95_high": 0.0, "ci95_low": 0.0, "requests": 80}, "per_rank_load_mae": {"beneficial_delta": -0.00016939720229623364, "ci95_high": 3.208854216084157e-05, "ci95_low": -0.00037278289889048494, "requests": 80}, "rank_distribution_cosine": {"beneficial_delta": -3.401271558988173e-05, "ci95_high": 1.0401523041182602e-06, "ci95_low": -7.010308990086582e-05, "requests": 80}, "rank_distribution_jsd": {"beneficial_delta": -4.53487038182018e-05, "ci95_high": 2.8119936104299606e-08, "ci95_low": -9.466269998948953e-05, "requests": 80}}`
- P_separated vs P_full: `{"active_expert_set_recall": {"beneficial_delta": 0.0, "ci95_high": 0.0, "ci95_low": 0.0, "requests": 80}, "decode_expert_top8_recall": {"beneficial_delta": -6.510416666666574e-05, "ci95_high": 0.0, "ci95_low": -0.00016276041666666435, "requests": 80}, "expert_histogram_cosine": {"beneficial_delta": 0.00033122503759807075, "ci95_high": 0.00086865293388495, "ci95_low": -0.00016948524538879352, "requests": 80}, "expert_histogram_jsd": {"beneficial_delta": 0.0002531610859035444, "ci95_high": 0.0009773727467664874, "ci95_low": -0.00045237670286833606, "requests": 80}, "hottest_rank_accuracy": {"beneficial_delta": 0.0, "ci95_high": 0.0, "ci95_low": 0.0, "requests": 80}, "per_rank_load_mae": {"beneficial_delta": 0.00012292869312494135, "ci95_high": 0.0003142196908247265, "ci95_low": -7.210437938910246e-05, "requests": 80}, "rank_distribution_cosine": {"beneficial_delta": 1.9787194140299945e-05, "ci95_high": 5.6072948103197244e-05, "ci95_low": -1.6080937609773657e-05, "requests": 80}, "rank_distribution_jsd": {"beneficial_delta": 2.2154099933930826e-05, "ci95_high": 6.650897619927423e-05, "ci95_low": -2.3442961224689173e-05, "requests": 80}}`
