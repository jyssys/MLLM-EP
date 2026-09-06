# Wait-aware tail PoC

`sitecustomize.py` in `online_routing_geometry/hooks` is the local observer.
It records DeepEP stage CUDA events, previous-event presence, closest event-wait
spans, and optional relevant communication-stream drains.  Raw traces are kept
under the timestamped result directory; analysis is performed by
`analyze_wait_aware_tail.py`.
