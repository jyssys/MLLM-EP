# Final decision

## `NO-OVERLAP-SIGNAL`

No exact candidate exceeds 5% credible direct request E2E headroom.

The system does contain legal concurrency, but the useful regions split into
three losing categories:

1. **Within-request and cheap:** routed/shared overlap has only 2.98--3.50% total
   mass.
2. **Within-request and large enough to test:** remote dispatch/local expert and
   dispatch/attention are slower under concurrency because communication is not
   resource-free.
3. **Actually positive:** complete-wave pipelines save 0.106--0.186 ms/layer,
   but require independent waves, have only a 4.35--4.59% sampled-median service
   upper (6.73% extreme upper), and collide with mature generic overlap work.

Refinement state changes absolute operation sizes and cross-state overlap
efficiency, but communication fraction changes only modestly and no phase-aware
complete pipeline beats the generic three-stage schedule. There is therefore no
clean dLLM-specific successor in this search space.

Recommended action: do not implement a production overlap scheduler for this
candidate. If revisited under a different runtime, require either (a) a native
multi-wave online workload with measured SLO-goodput headroom above 8%, or (b) a
new exact same-request dependency break not already covered by StreamEP/X-Stage.
