import os

if os.environ.get("MODPHASE_ENABLE", "0") == "1":
    from poc_modality_phase_pivots.pairwise_hook import install

    install()
