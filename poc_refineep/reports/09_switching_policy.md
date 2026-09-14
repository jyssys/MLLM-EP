# Switching policy

No runtime selector was implemented.

For the 25 real nonempty compacted shapes, `low_latency < normal_fresh` in
25/25 cases. The same holds for 15/15 controlled shapes spanning M=32/128/512,
fanout 1/2/4, remote fraction 0/0.75/1, and balanced/skewed load. Thus the O0
lookup table degenerates to:

```text
if compacted routed work is nonempty:
    use existing low_latency
else:
    skip
```

This is an existing-path deployment recommendation, not a research method.
The selector has no learned or dynamic contribution on this corpus, and no
overhead measurement is needed for a constant choice.

At the two largest real cases, an exact cached normal route handle is slightly
faster than LL, but routes change across real refinement steps. Treating that
future-known same-handle diagnostic as a deployable selector would violate the
correctness contract.
