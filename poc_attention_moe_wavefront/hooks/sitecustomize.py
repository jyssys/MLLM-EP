import os

if os.environ.get("WAVEFRONT_ENABLE") == "1":
    from poc_attention_moe_wavefront.instrumentation import install

    install()

