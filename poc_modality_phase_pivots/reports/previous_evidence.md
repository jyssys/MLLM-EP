# Existing PoC evidence consulted before launch

All entries below were used only to frame controls; final decisions require the
fresh measurements in this PoC.

| Existing artifact | Relevant result | Constraint imposed here |
|---|---|---|
| `poc_flashvep/reports/qwen3vl_full_serving_resource_atlas.md` | Whole vision encoder block concurrent with Dispatch/Combine/Expert slowed wall time by 12.4%/5.0%/8.9% | Test the distinct matched-shape *language Attention over vision tokens* question; do not recycle encoder overlap as new evidence |
| `poc_flashvep/reports/live_causal_modality_wavefront.md` | Token-split wavefront reached high timeline overlap but was 7.42x slower | Never split token axes or count overlap without split tax |
| `poc_flashvep/reports/modality_aware_moe_granularity_poc.md` | Text and vision both selected M=512 | Require fresh phase-policy curves; do not assume modality granularity |
| `online_moe_ep_runtime_discovery_20260905_191711/analysis_final/communication_sms_summary.csv` | Prior sms8/12/20 differences were small for M=1 and M=284 | Include a denser 4/8/12/16/20 sweep with actual modality routes, but promote only on best-static regret |
| `poc_flashvep/reports/attention_moe_wavefront_poc.md` and prior branch | Attention→MoE token wavefront oracle was already screened | This PoC allows only full request/layer units and cross-request scheduling |

The new causal control fixes Attention token count at 2,363 and fixes the
pairwise EP input/route across text and vision. This separates content modality
from execution shape, which prior artifacts did not directly isolate.
