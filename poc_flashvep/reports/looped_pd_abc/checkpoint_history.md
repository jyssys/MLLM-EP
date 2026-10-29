# Git checkpoints

These are intermediate checkpoints, not completed experiment decisions.
The active GPU campaign continues after each checkpoint.

| Commit | Contents | Push check |
|---|---|---|
| `7b8bffa636c27724d7d20be3e6aba4e6e0401731` | Initial tested prototypes, early evidence, and explicit draft reports | Remote branch SHA matched after push |
| `ee2172e250fc66d44787fd09a82764237cf37420` | Frozen prototypes, completed reference limits, completed-run evidence, and in-progress serving tables | Remote branch SHA matched after push |
| `2b31a247799d1f289c387cd2de4ce9c824ea0608` | Foreground evidence-I/O defect, tested background recorder, canonical retiming, and explicit draft warnings | Remote branch SHA matched after push |

Branch: `flashvep/looped-pd-abc-disaggregation-poc`.
Remote: `https://github.com/jyssys/MLLM-EP.git`.

The second checkpoint excluded active request logs and mutable controller state.
It committed exact staged paths from the two owned PoC directories only.
The unrelated staged-index hash remained `c162445edb25097abbf18627e41ba8dffc2a15f9a35e6d11385b37a9eb449a22`.
Later completed runs and analysis updates still require a final scoped commit.
The final report must identify its actual final commit and remote verification separately.

The third checkpoint contains tested code and timing-correction reports.
It does not contain the active serving logs or a final architecture verdict.
The outside staged-index hash remained unchanged after this push.
