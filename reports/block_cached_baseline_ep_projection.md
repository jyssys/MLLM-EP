# Block-cached baseline EP projection

Matched GSM8K-32 trajectories; simulated routed-MoE stage only. B1 includes a conservative full-prefix refresh at every new block shape.

| Target | B0 naive ms/request | B1 cached ms/request | stage gain | fresh pair reduction |
|---|---|---|---|---|
| EP4 | 2002.752 | 1145.246 | 42.816% | 90.841% |
| EP8 | 1743.052 | 1190.366 | 31.708% | 90.841% |
