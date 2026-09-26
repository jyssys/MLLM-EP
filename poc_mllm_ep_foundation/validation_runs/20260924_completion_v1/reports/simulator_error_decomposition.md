# Simulator error decomposition

The replay feature contract was corrected from source-row sums to destination
received assignments, and request targets were changed from a sum of unrelated
per-layer medians to the median of complete-repeat totals. High-repeat anchors
(2 warm-ups, 9 measurements) and a component-wise stage option were selected
on development only. Frozen selection chose
`aggregate_layer_knn_16` for EP2 and
`componentwise_stage_models` for EP4. The new untouched v4
test still misses the operating gate. Physical repeat variance and DeepEP
small-message dispatch/combine plateaus dominate the remaining unexplained
tail; arbitrary per-request residual fitting was not used.

- EP2 final request repeat CV median/P90: 0.072 / 0.083
- EP4 final request repeat CV median/P90: 0.116 / 0.246
- EP2 signed request bias: 6.840 ms
- EP4 signed request bias: 11.461 ms
