# Source discovery pass 3: model and request contracts

The model snapshot config records 48 MoE layers, 128 routed experts, top-8,
hidden size 2048 and expert intermediate size 768.  The serving path keeps a
linear expert map (32 local experts per physical worker in TP2/DP2/EP4), and
the Qwen-VL multimodal path presents vision embeddings to the same transformer
execution contract as text.

The key hidden choices are semantic rather than syntactic: top-k is treated as
immutable after routing; all selected contributions are combined before the
next block; one scheduler step can contain mixed phases; and response
completion order is observable at the API but not a model dependency.

**Assumptions generated:** A12--A14, A25--A29, A43--A50.

Existing speculative/partial-completion and modality experiments are negative
controls.  They show that relaxing a semantic contract can have attractive
operator-level numbers while failing quality, verification, or request-level
headroom.  Those contracts remain in the catalog but are not promoted.
