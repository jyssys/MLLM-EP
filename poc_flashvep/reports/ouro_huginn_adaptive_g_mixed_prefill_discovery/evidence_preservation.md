# Evidence preservation and validation

The final raw audit verifies 244 successful worker attempts and one preserved failed attempt.
The failure is `dp_ouro_w1_hybrid_s0_gpu0`.
Its source diff and traceback remain available.
The repaired DP retry does not overwrite it.

The portable archive contains 2,866 original files in 246 lossless XZ archives.
The compressed size is 19,313,160 bytes.
The audit checks every archived file against its SHA256.
The original raw files remain in place.
The archives contain no model weights.

`raw_archive_manifest.json` maps each original path to its archive and hash.
`evidence_hashes.csv` provides the same file mapping as a table.
The large batch-plan and ITL CSV files reside in the large-derived-CSV archive.
Worker logs, raw outputs, source snapshots, and the failure log reside in their attempt archives.

The checkpoint integrity record contains five actual safetensors-file hashes.
The hashes match the local Hugging Face LFS object identifiers.
The original checkpoint files remain outside Git.

The final CPU test record reports 14 passing tests and seven deselected tests.
The deselected tests do not become successful GPU evidence.
The separate live checkpoint probes supply the GPU numerical evidence.

The audit also checks the unchanged hash of the unrelated staged diff.
This campaign does not reset, clean, or commit the user's unrelated work.
The final response supplies the committed SHA and the verified remote branch SHA.

The report lint applies structural clarity checks to new report prose only.
Its lexical synonym scan excludes domain collisions such as fixed depth versus a bug fix.
Frozen specification snapshots and raw evidence remain unchanged.
