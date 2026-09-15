# 05 — Future EP-cost prediction

The sampled-layer current-to-next position-level rank-vector cosine medians
are 0.9762 for GSM8K and 0.9729 for partial HumanEval. This indicates
coarse rank-level temporal structure and would make H0 current-route or H1
EMA plausible *features* if a physical oracle were strong.

No held-out model relating token-level rank cost to actual next-wave EP wall
time was fit; C4 measured rank service time and C5 counterfactual latency are
not established. The previous matched-composition EP4 replay showed why a
stable rank vector is insufficient by itself: physically complementary
signatures did not materially beat random composition at the best-static
scale. Shuffled-history, random-cost, and confidence-only policy controls
were not run because no policy was promoted.

`EP_COST_PREDICTION.csv` is labeled stability diagnostic, not predictor
validation. The question "can current route predict future physical saving?"
remains untested as a latency question; it cannot rescue a sparse near-tie
set and an unresponsive physical execution path.
