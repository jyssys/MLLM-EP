# EP-aware refresh scheduler decision

**Not implemented by gate.**

The scheduler required all three links:

1. future-dormant routed work is large;
2. stale/periodic reuse preserves the full trajectory;
3. expert/EP features identify a safe, expensive subset beyond semantic cues.

Only the first link is partially true.  H=1 has 8.35% mean optimistic mass,
but its fixed-floor perfect ceiling is 7.11%; causal reuse changes trajectories;
and the deployable joint proxy projects 2.37%/1.30% E2E.  Route-safe refresh
projects 0.63%/0.54%.

Therefore no custom variable-row runtime, cache manager, or scheduler was
built.  The diagnostic instrumentation and reproducible plans are retained for
future models where the perfect post-Epoch ceiling might be larger.
