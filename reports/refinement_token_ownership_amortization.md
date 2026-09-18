# Refinement-coupled ownership amortization

M0 is free logical remap, M1 moves one BF16 hidden row (4096 bytes), and M2 conservatively moves 20 layer states (81920 bytes). Migration uses the EP2-calibrated endpoint model and is charged once per moved token/block. These are sensitivity scenarios, not production migration measurements.

```json
{
  "ep4": {
    "M0_zero": {
      "state_bytes_per_token": 0,
      "moved_tokens": 21252,
      "migration_bytes": 0,
      "migration_latency_ms": 0.0,
      "repeated_net_stage_gain_percent": 0.0004205904996982448,
      "ar_one_shot_net_stage_gain_percent": 0.00011266863181760483,
      "dllm_repeated_break_even": true,
      "ar_one_shot_break_even": true,
      "minimum_equivalent_future_refinements": 0.0
    },
    "M1_hidden_row": {
      "state_bytes_per_token": 4096,
      "moved_tokens": 21252,
      "migration_bytes": 87048192,
      "migration_latency_ms": 111.02886608580553,
      "repeated_net_stage_gain_percent": -0.035827061340917635,
      "ar_one_shot_net_stage_gain_percent": -0.03613498320879827,
      "dllm_repeated_break_even": false,
      "ar_one_shot_break_even": false,
      "minimum_equivalent_future_refinements": 321.71910900006213
    },
    "M2_20_layer_state": {
      "state_bytes_per_token": 81920,
      "moved_tokens": 21252,
      "migration_bytes": 1740963840,
      "migration_latency_ms": 130.37161605617158,
      "repeated_net_stage_gain_percent": -0.042141899015870225,
      "ar_one_shot_net_stage_gain_percent": -0.04244982088375087,
      "dllm_repeated_break_even": false,
      "ar_one_shot_break_even": false,
      "minimum_equivalent_future_refinements": 377.7669865066911
    }
  },
  "ep8": {
    "M0_zero": {
      "state_bytes_per_token": 0,
      "moved_tokens": 23894,
      "migration_bytes": 0,
      "migration_latency_ms": 0.0,
      "repeated_net_stage_gain_percent": 0.006025503057592531,
      "ar_one_shot_net_stage_gain_percent": 0.0003628020809135069,
      "dllm_repeated_break_even": true,
      "ar_one_shot_break_even": true,
      "minimum_equivalent_future_refinements": 0.0
    },
    "M1_hidden_row": {
      "state_bytes_per_token": 4096,
      "moved_tokens": 23894,
      "migration_bytes": 97869824,
      "migration_latency_ms": 111.15542642296658,
      "repeated_net_stage_gain_percent": -0.03621055425346609,
      "ar_one_shot_net_stage_gain_percent": -0.041873255230145116,
      "dllm_repeated_break_even": false,
      "ar_one_shot_break_even": false,
      "minimum_equivalent_future_refinements": 116.41624878421747
    },
    "M2_20_layer_state": {
      "state_bytes_per_token": 81920,
      "moved_tokens": 23894,
      "migration_bytes": 1957396480,
      "migration_latency_ms": 132.9028227993984,
      "repeated_net_stage_gain_percent": -0.04447397700555136,
      "ar_one_shot_net_stage_gain_percent": -0.050136677982230385,
      "dllm_repeated_break_even": false,
      "ar_one_shot_break_even": false,
      "minimum_equivalent_future_refinements": 139.1929173503917
    }
  }
}
```

The mandatory AR control observes one execution and benefits for only the next refinement. The repeated dLLM horizon is compared with exactly the same setup cost. The final summary reports whether each sensitivity breaks even.
