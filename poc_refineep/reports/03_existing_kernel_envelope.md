# Existing DeepEP envelope

## Protocol

The 25 real route shapes were measured in five independent process restarts.
Each restart used 8 warmups and 30 timed repeats per policy, randomized policy
order, identical input/route, and critical-rank same-GPU CUDA events. The
policies were:

- `normal_fresh`: layout + normal dispatch + weight semantics + combine;
- `normal_cached`: exact same-route handle reuse, an optimistic diagnostic for
  repeated metadata, not a deployable changing-route path;
- `low_latency`: legacy BF16 low-latency dispatch/combine through the valid
  intranode NVLink/P2P fallback.

## Real routes

| shape | normal fresh (ms) | cached handle (ms) | low latency (ms) | LL gain vs normal |
|---|---:|---:|---:|---:|
| very-small | 0.2449 | 0.2036 | 0.0909 | 62.66% |
| small | 0.2457 | 0.2054 | 0.0903 | 63.23% |
| medium-small | 0.2492 | 0.2084 | 0.0928 | 62.74% |
| medium | 0.2610 | 0.2212 | 0.1292 | 50.48% |
| large | 0.2768 | 0.2311 | 0.2125 | 24.22% |

Low-latency beats `normal_fresh` in 25/25 cases and in every one of five
restarts. The paired restart median gain ranges from 57.4% to 62.7%; the
minimum individual large-shape gain is 1.87%, but its winner never reverses.
Absolute restart latency has a 27--34% CV because node state varied, so the
paired policy ordering—not an unpaired aggregate—is the primary evidence.

Cached normal beats LL only in the two largest cases (M=802/1024) and cannot be
used as the best-existing oracle for arbitrary changing routes. Even there LL
still beats the normal path that rebuilds the valid route contract.

## Controlled routes

Low-latency also beats normal in all 15 matched controls. At M=32 the gain is
61.8--63.9%, at M=128 it is 56.2--63.7%, and at M=512 it is 18.0--61.2%.
Fanout, remote fraction, and skew change the magnitude but do not expose a
third region where normal wins.

## Meaning

There is no uncovered high-mass medium/small regime between the existing
endpoints. The best current path for every nonempty real compacted shape is the
same low-latency path, so O0 is a static choice; dynamic mode switching itself
has no per-shape selector value on this corpus.
