# Benchmark quality

Status: `NOT_RUN` for candidate GSM8K and MATH-500 accuracy.
No new loop-aware serving candidate passed the incremental-method gate.
The standalone distributed operator does not generate model answers.
Its numerical checks cannot establish benchmark accuracy.

The model remains the original BF16 Base checkpoint at Static R4.
The native baseline records actual generated token IDs and exit-depth arrays.
Bounded hardware runs ignore EOS to retain constant measurement length.
Separate EOS-enabled probes permit natural termination but retain a token cap.
Neither lane is a GSM8K or MATH-500 accuracy evaluation.

The exact layout control compares actual autoregressive outputs under matched inputs.
Its sampled KV checks test initialized cache values.
These checks do not replace full-answer evaluation.
There is no claim of population-level quality equivalence.

If a later integrated candidate qualifies, use the same original prompt and sampling protocol.
Start with matched GSM8K 64 and MATH-500 50 questions.
Record correct/total, parse failures, EOS, truncation, and paired confidence intervals.
Expand only after the numerical and full-generation checks pass.
