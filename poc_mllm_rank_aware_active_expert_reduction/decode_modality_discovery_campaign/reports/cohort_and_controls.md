# Frozen cohorts and controls

- Discovery: **32** requests; confirmation: **24** fresh requests.
- Image hashes are unique and split-disjoint; prior Q2-2/Q2-3 hashes were excluded before any route/damage inspection.
- Discovery domains: `{'dense_relation': 8, 'docci': 6, 'nocaps': 6, 'scene_layout': 6, 'textcaps': 6}`.
- Confirmation domains: `{'dense_relation': 6, 'docci': 6, 'nocaps': 4, 'scene_layout': 4, 'textcaps': 4}`.
- Conditions: original image, same-domain token-count-matched image swap, and semantic-neutral same-geometry image.
- Question and teacher-forced token IDs are fixed across conditions. Layers `[4, 12, 24, 36, 44, 47]` and steps `[8, 16, 32, 64, 96, 112]` were frozen in advance.
- Confirmation was frozen before primary selection: `True`.
