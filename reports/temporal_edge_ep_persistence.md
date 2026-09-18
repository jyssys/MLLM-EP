# Temporal-Edge EP: exact live-MASK persistence

Evidence: exact current-block token/layer/expert identities from threshold-.95 GSM8K-128. Requests 0--63 are discovery and 64--127 held out. EP4/EP8 ownership is simulated; no quality or timing claim is made here.

## Primary result

| split | STAY route count | STAY router mass |
|---|---:|---:|
| discovery | 68.316% | 72.406% |
| held-out | 69.305% | 73.319% |
| all | 68.821% | 72.872% |

ENTER and EXIT are each the complement of STAY because top-k remains eight. Full stratification by layer, refinement, mask ratio, confidence, slot, weight decile, block index and EP-local/remote destination is in the machine summary. The discovery-selected subgroup was frozen before held-out evaluation: `slot=1`; discovery/held-out STAY were 87.73%/88.28%.

## Negative controls

```json
{
  "adjacent_live_mask": {
    "pairs": 258476,
    "routes": 30810096,
    "stay_route_fraction": 0.6882082418698079,
    "stay_router_mass_fraction": 0.7287212157993802
  },
  "decoded_stable_positive_control": {
    "pairs": 258476,
    "routes": 25623704,
    "stay_route_fraction": 0.9251343209397049,
    "stay_router_mass_fraction": 0.947696713874463
  },
  "matched_random_nonadjacent_refinement": {
    "pairs": 240027,
    "routes": 27006144,
    "stay_route_fraction": 0.5254376189358985,
    "stay_router_mass_fraction": 0.565235567625156
  },
  "same_layer_different_token": {
    "pairs": 258476,
    "routes": 29033672,
    "stay_route_fraction": 0.5751778831144748,
    "stay_router_mass_fraction": 0.6173659438955824
  },
  "same_token_different_block": {
    "pairs": 205827,
    "routes": 23708048,
    "stay_route_fraction": 0.3724052271194997,
    "stay_router_mass_fraction": 0.3984888637706706
  },
  "shuffled_expert_rows_preserving_layer_popularity": {
    "pairs": 129238,
    "routes": 10262640,
    "stay_route_fraction": 0.2333708480468963,
    "stay_router_mass_fraction": 0.25089789838785265
  }
}
```

Run length P50/P75/P90/P95/P99/max is 1/3/6/8/15/32 refinements. Adjacent repeated refinement is materially stronger than the controls, so the structural dLLM-specific edge signal is real.
