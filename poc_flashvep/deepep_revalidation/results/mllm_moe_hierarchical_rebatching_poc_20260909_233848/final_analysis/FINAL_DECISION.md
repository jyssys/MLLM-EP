# Final decision

**NO_HIERARCHICAL_HEADROOM**

The central three-way compatibility conflict does not reproduce. Attention and
MoE select the same simple grouping in every observer restart. The full exact
hierarchy has a median optimistic direct-BCT oracle of 1.37% (maximum 6.42%)
and a median favorable feasible lower-bound oracle of 1.35%. Its MoE-specific
increment over independent vision/LM stage batching is only 0.50%, and its
vision increment over BatchGen-style LM-only rebatching is 1.37%.

Both required specificity gates and the 8% implementation gate fail. No
prototype or Kimi run was performed.
