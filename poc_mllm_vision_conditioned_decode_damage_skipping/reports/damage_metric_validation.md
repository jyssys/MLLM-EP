
# Damage metric validation

Primary metric was frozen before functional results:

`max(0, target_logprob_drop_nats) + 2*JSD + 1[top1_flip]`.

It is based on actual next-token logits after the remaining forward, while local
mixture relative-L2/cosine remain broad-coverage secondary labels. Functional
score mean/P50/P80/P90/P99 were 0.0247063 / 9.77427e-06 /
0.00496187 / 0.0370676 / 0.681006. Top-1 flip rate was
1.17%. Local-mixture vs functional-damage
Spearman was 0.0136. Local mixture relative-L2 P50/P90/P99
were 0.2266 /
0.5243 /
0.8062.

No metric was selected after predictor results, and no natural-generation task
quality claim is made from these local/one-step labels.
