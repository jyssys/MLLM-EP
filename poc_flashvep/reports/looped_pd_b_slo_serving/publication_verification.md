# Publication verification

The compact scientific snapshot is `70dd6d82881c130f1f52c6b3ec0bd4d9d6ef5fe9`.
The remote branch is `flashvep/looped-pd-b-slo-serving-poc` in `jyssys/MLLM-EP`.
The publication helper verified that the remote branch SHA equals that snapshot SHA.
It also checked the required final reports, component CSV, sensitivity CSV, and compressed dataset in that commit tree.

The initial unpublished commit is `be05ecba9ae4074709ab4655bf7600742705e719`.
GitHub rejected its126,308,939-byte `W1_long.json` file.
That commit remains in the local archive reference:

`refs/heads/flashvep/looped-pd-b-slo-serving-local-unpublished-be05ecba9ae4`.

No force push was used. The remote branch did not exist before compact publication.
The publication commit has the original frozen experiment commit as its parent.
The existing local raw JSON remains unchanged and available.
The remote gzip reconstructs the original bytes exactly. Read `configs/dataset_storage.json`.

The unrelated staged-index SHA remains:

`fa76eff21de6acef60a0cce13ae412109ddc43998300cd0afeafded5794e3dfa`.

Ten CPU contract tests pass. Saved-evidence validation reports184 runs and zero integrity issues.
The184 runs include167 COMPLETE and17 FAILED records. This is not a claim that every experiment passed.
No new GPU inference started during publication closeout. Owned burners are restored on GPUs0–7.

This verification-only commit can be a child of the verified scientific snapshot.
The final response supplies the current HEAD SHA and its remote verification.
Do not interpret the initial supervisor push-failure log as the result of this later successful publication.
