# Selected cross-depth attention-pattern certificate

## Result and scope

The selected native BF16 keys yield full-rank cross-depth certificates for 36/36 tested pairs in each Base model.
This is an exact-real algebraic result on observed tensors.
It is not a GPU-rounded softmax theorem, a measured cache-hit rate, or a latency result.

Each model contributes one controlled TP1 request.
The fixture selects three physical layers and three heads.
It covers depth pairs (1,2), (2,3), (3,4), and (1,4).
Every pair uses the same 257 visible key positions and 128 features per head.
No values or actual query vectors enter this test.

## Equation

Let `D_a(j)=K_a(j)-K_a(0)` and `D_b(j)=K_b(j)-K_b(0)`.
Identical exact-real softmax patterns require matching score differences.
The common nonzero head scale cancels from this condition.
For the selected positions, the necessary condition is:

```text
[D_a, -D_b] [q_a; q_b] = 0.
```

The combined matrix has 256 rows and 256 columns.
The computation maps the exact dyadic BF16 words into the odd-prime field with p=65521.
Each reported nonzero determinant proves a nonzero rational determinant.
The matrix therefore also has full rank over the real numbers.
Its only null vector is zero.

Thus exact-real pattern equality on these selected positions requires both query vectors to be zero for each certified pair.
Equality over the complete visible prefix would also require equality over these selected score differences.
The result does not assert that an actual observed query is zero or nonzero.
The test captures no query distribution.

## Limits

Finite-precision score accumulation and softmax rounding can create coincidences that exact-real arithmetic does not predict.
This certificate does not reject those coincidences.
Different attention patterns can also produce the same weighted output for particular values.
The certificate does not reject that possibility.
It does not test approximate reuse, numerical tolerance, benchmark accuracy, or any candidate runtime.

The result weakens an unqualified exact cross-loop pattern-reuse premise in this bounded fixture.
It does not prove a universal impossibility result for every Ouro tensor or representation.

## Reproduction

The script is `scripts/cross_depth_key_rank.py`.
The versioned results are `artifacts/cross_depth_key_rank_base14_v1.json` and `artifacts/cross_depth_key_rank_base26_v1.json`.
They preserve input hashes, model revisions, selected pairs, ranks, and determinant residues.
The original word-to-field checks remain in the earlier per-head rank artifacts.
No GPU rerun supplies this analysis.
