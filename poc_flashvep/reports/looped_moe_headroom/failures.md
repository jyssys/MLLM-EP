# Failures and unresolved limits

## Environment preparation

The initial Python 3.14 installation could not build the pinned `tiktoken==0.8.0` dependency.
The experiment then used an isolated Python 3.12 environment.
The old failed environment remains separate.
No model or checkpoint changed.

The first native extension builds lacked an executable Ninja on PATH.
Two owned build trees were stopped before a parallel build.
The final official extension build completed for SM90.
The logs preserve the failed and interrupted attempts.

FLA imported an inherited TileLang installation with an incompatible TVM-FFI version.
A minimal `import tilelang` reproduced the failure.
The validated local pins use TileLang 0.1.15 and TVM-FFI 0.1.12.
A native import regression test then passed.
The CUDA DSL package also needed its standard package-path registration in the isolated environment.
These changes affected dependencies, not model source.

## Loader and probe corrections

The first Foil load lacked the root remote-code files in its local download.
The corrected download includes those official files and the Foil-1 subfolder.
The tokenizer call now explicitly trusts the pinned remote code.

An early probe revision referenced a reuse-error variable at the wrong scope.
Its failed run remains in the raw archive.
The corrected probe passed the native no-op and q=0 checks.

## Backend numerical difference

The original whole-model comparison used a relative-L2 threshold of 0.05.
FA4 versus BF16 explicit attention exceeded it for both IFM models.
Ouro measured 0.0574008115.
Huginn measured 0.0618398562.
Those failures remain unchanged.

A separate actual-Q/K/V diagnostic checked the attention operator against an FP32 reference.
It also checked eight generated tokens.
These independent checks do not turn the earlier test into a pass.
The later strict reference disables TF32 during its diagnostic matrix multiplications.
The native TF32 policy is restored after each reference calculation.

No FA3-versus-FA4 benchmark quality equivalence claim is available.
The model comparisons use the same validated FA4 path on both sides.

## Execution and measurement limits

The first follow-up dispatcher reached the lease before burner restoration finished.
Its nonblocking lock failed before it launched a timing worker.
The dispatcher now serializes campaigns with a blocking lease.
The successful retry preserves the earlier launcher error.

Foil's shorter observer runs exceed the 10% observer-overhead threshold.
Those timing fractions remain PARTIAL.
The optimistic clean-denominator bound remains explicitly analytical.

The first Foil operator microbenchmark included host offset staging.
The corrected v2 benchmark prepares GPU offsets before timing.
The original raw results remain available.

Only one recurrent site per model enters the operator microbenchmark.
Hardware instruction counts and HBM bandwidth were not measured.
Removed input rows can leave grouped-kernel padding and launch counts unchanged.
No achieved candidate E2E acceleration is inferred from row removal.

The downstream pilot is small and uses a 256-token cap.
Its strict parser is not the official MATH equivalence evaluator.
Weak baselines, failed parsing, and capped reasoning can limit interpretation.
The report does not certify a 1 percentage point accuracy margin.

All runtimes in this study lack a generation KV cache.
The results do not represent optimized cached-decode serving.
B2/B4, production serving, and multi-GPU model parallelism were not evaluated.

## Final evidence checks

The stricter FP32-reference follow-up changed token 4 of eight for both IFM models on a different prompt.
Its attention-operator check still passed.
The report does not interpret that operator pass as whole-model generation parity.

The first independent-site Foil control jobs failed while serializing loader metadata.
The loader returned empty sets, which JSON cannot encode directly.
The adapter now converts those sets to sorted lists.
Eight new run IDs completed the same controls.
A separate strict-load run also passed.
The eight failed run directories remain preserved.

The first workload reconstruction script expected a MATH parquet file.
The pinned source instead contains test.jsonl.
The corrected reader reproduced all 508 frozen text hashes.
The failed command log remains available.

The first final integrity check required top-1 agreement to equal exactly 1.0.
Fourteen no-op rows instead contained 0.9999999403953552 from CUDA FP32 mean reduction.
A GPU-only all-true reduction reproduced the exact error without loading a model.
The revised integrity check permits one FP32 epsilon only for that legacy aggregate.
NLL, KL, and the native exact-logit criteria remain unchanged.
New metric code calculates the ratio from integer counts.

The Python unittest discovery command found no class-based tests in the function-based fixture module.
The actual module entry point then executed all 13 fixtures successfully.
The zero-test command is not counted as a passed test suite.
