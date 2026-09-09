#!/usr/bin/env python3
"""Native PCP support smoke; failures are classified as environment/support."""
import argparse
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--pcp", type=int, required=True)
    args = p.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "4,5,6,7"
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model, dtype="bfloat16", tensor_parallel_size=1,
        prefill_context_parallel_size=args.pcp, trust_remote_code=True,
        enforce_eager=True, kv_cache_memory_bytes=2 << 30,
        max_model_len=8192, max_num_batched_tokens=8192, max_num_seqs=8,
        enable_prefix_caching=False, enable_flashinfer_autotune=False,
    )
    print(llm.llm_engine.vllm_config.parallel_config)
    output = llm.generate(
        ["Context parallelism correctness test."],
        SamplingParams(temperature=0, max_tokens=2), use_tqdm=False,
    )[0]
    print(list(output.outputs[0].token_ids))
    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
