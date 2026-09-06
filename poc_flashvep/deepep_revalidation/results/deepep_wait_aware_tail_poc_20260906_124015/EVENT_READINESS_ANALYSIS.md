# Event readiness analysis

The installed DeepEP EventHandle has no nonblocking query method. `previous_event_present` was therefore recorded directly and the table below uses same-device CUDA timing around the stock `EventOverlap.current_stream_wait()` as a closest wait proxy. It is not a cross-GPU absolute readiness clock.

- invocations: 695,232

- previous-event object present: 0 (0.000%)

- event-wait median/p99/max: 0.0220 / 1.1515 / 8.7961 ms


| cutoff_ms | class | n | dispatch_p50_ms | dispatch_p99_ms | tail_gt10_rate_pct | tail_gt20_rate_pct |
|---|---|---|---|---|---|---|
| 0.05 | READY_PROXY | 458860 | 0.1013759970664978 | 1.1703538835048544 | 0.045765592991326334 | 0.03203591509392843 |
| 0.05 | NOT_READY_PROXY | 236372 | 0.6418560147285461 | 4.412512922286989 | 0.3363342527879783 | 0.1772629583876263 |
| 0.1 | READY_PROXY | 503535 | 0.10400000214576721 | 1.2258111763000406 | 0.05242932467455092 | 0.03713743831114024 |
| 0.1 | NOT_READY_PROXY | 191697 | 0.7779200077056885 | 5.080910129547141 | 0.3865475203054821 | 0.19770784101994293 |
| 0.5 | READY_PROXY | 668793 | 0.1233920007944107 | 1.5592639446258545 | 0.05741686889665413 | 0.04171694380772526 |
| 0.5 | NOT_READY_PROXY | 26439 | 1.3077759742736816 | 43.65438888549663 | 2.3488029047997276 | 1.0855176065660577 |
| 1.0 | READY_PROXY | 687225 | 0.12598399817943573 | 1.68230402469635 | 0.06824548001018589 | 0.04801920768307323 |
| 1.0 | NOT_READY_PROXY | 8007 | 1.6997120380401611 | 756.7953747558504 | 6.694142625202948 | 2.947421006619208 |

## Stock-only predictive check

The policy-pooled table above is descriptive.  Restricting to the three stock
runs (160,704 decode invocations) gives the primary check:

| proxy cutoff | ready rows | not-ready rows | P(dispatch >10 ms | ready) | P(dispatch >10 ms | not-ready) | relative risk |
|---:|---:|---:|---:|---:|---:|
| 1.0 ms | 159,660 | 1,044 | 0.050% | 10.441% | 208.4× |
| 0.5 ms | 155,592 | 5,112 | 0.039% | 2.504% | 63.9× |

The event-wait proxy is therefore predictive, but it is a downstream
same-device measurement; `previous_event_present` remained false because DBO
was disabled and the installed C++ binding has no event query method.
