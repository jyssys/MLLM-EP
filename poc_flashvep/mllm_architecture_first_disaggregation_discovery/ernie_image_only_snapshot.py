"""Build an offline image-only ERNIE overlay over verified publisher files."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path


PUBLISHER_DIGESTS = {
    "configuration_ernie4_5_vl.py": "379cd61ce04eb9a18b8370a7771d25b96c96d1a9ba1fef7181bc5633fb5877b8",
    "modeling_ernie4_5_vl.py": "69a72e4734678d8f2a64c163839464128bebf434c78a1d81541dbd8b57ffd921",
    "processing_ernie4_5_vl.py": "1c2b5c2237d1889b90790fa4a8518eede0fa25731243dad966f042cd4e8870f8",
}
PROCESSOR = "processing_ernie4_5_vl.py"


def image_only_processor(source):
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    imports = [node for node in tree.body if isinstance(node, ast.Import) and
               [alias.name for alias in node.names] in (["decord"], ["requests"])]
    font = [node for node in tree.body if isinstance(node, ast.If) and "Roboto-Regular.ttf" in ast.unparse(node)]
    video = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VideoReaderWrapper"]
    media = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.unparse(node.func) == "requests.get" and
             ast.unparse(node) == "requests.get(url)"]
    if list(map(len, (imports, font, video, media))) != [2, 1, 1, 1]:
        raise ValueError("Publisher processor structure changed")
    replacements = [(node, []) for node in imports + font]
    replacements.append((video[0], ["class VideoReaderWrapper:\n", "    def __init__(self, *args, **kwargs):\n",
                                     '        raise ValueError("Video is disabled in this image-only snapshot")\n']))
    replacements.append((media[0], [f'{" " * (len(lines[media[0].lineno - 1]) - len(lines[media[0].lineno - 1].lstrip()))}'
                                     'raise ValueError("Remote media is disabled in this offline snapshot")\n']))
    for node, replacement in sorted(replacements, key=lambda item: item[0].lineno, reverse=True):
        lines[node.lineno - 1:node.end_lineno] = replacement
    patched = "".join(lines)
    ast.parse(patched)
    return patched


def build(snapshot, output):
    if snapshot.resolve() == output.resolve():
        raise ValueError("The overlay directory must differ from the verified publisher snapshot")
    for name, expected in PUBLISHER_DIGESTS.items():
        if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Unreviewed source version for {name}")
    output.mkdir(exist_ok=True, parents=True)
    patched = image_only_processor((snapshot / PROCESSOR).read_text())
    overlay_processor = output / PROCESSOR
    if overlay_processor.is_symlink() or (overlay_processor.exists() and overlay_processor.read_text() != patched):
        raise ValueError("Existing processor differs from the planned overlay")
    if not overlay_processor.exists():
        overlay_processor.write_text(patched)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    names = {file.name for file in snapshot.iterdir() if file.is_file() and file.suffix in {".json", ".py", ".model"}}
    names.update(index["weight_map"].values())
    for filename in sorted(names - {PROCESSOR}):
        link = output / filename
        original = (snapshot / filename).resolve()
        if link.is_symlink() and link.readlink() == original:
            continue
        link.symlink_to(original)
    return {"publisher_source_sha256": PUBLISHER_DIGESTS,
            "overlay_processor_sha256": hashlib.sha256(patched.encode()).hexdigest(),
            "overlay_changes": ["Remove optional video runtime import and raise on video reader use",
                                "Remove import-time font request and reject remote media in publisher processor"],
            "scope": "Same publisher model, config, tokenizer, chat template, and local image path"}


def preflight(snapshot):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from PIL import Image
    from transformers import AutoTokenizer, __version__ as transformers_version
    from vllm import __version__ as vllm_version
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.processing.context import TimingContext
    from vllm.multimodal.processing.inputs import ProcessorInputs

    model = str(snapshot.resolve())
    config = ModelConfig(model=model, trust_remote_code=True, max_model_len=8192, dtype="bfloat16",
                         enforce_eager=True, limit_mm_per_prompt={"image": 2, "video": 0})
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=True)
    processor = MULTIMODAL_REGISTRY.create_processor(config, tokenizer=tokenizer)
    images = [Image.new("RGB", (256, 256), "red"), Image.new("RGB", (384, 256), "blue")]
    checks = []
    for count in range(3):
        message = [{"role": "user", "content": [{"type": "image"}] * count +
                    [{"type": "text", "text": "Describe the image."}]}]
        prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        media = processor.info.get_data_parser().parse_mm_data({"image": images[:count]} if count else {})
        packed = processor.apply(ProcessorInputs(prompt=prompt, mm_data_items=media), TimingContext(enabled=False))
        spans = [{"offset": span.offset, "length": span.length} for span in packed["mm_placeholders"].get("image", [])]
        if len(spans) != count or any(span["length"] <= 0 for span in spans):
            raise ValueError(f"Native processor did not preserve all {count} image placeholders")
        checks.append({"images": count, "packed_tokens": len(packed["prompt_token_ids"]), "image_spans": spans})
    return {"evidence": "CPU_PREFLIGHT", "vllm_version": vllm_version, "transformers_version": transformers_version,
            "resolved_native_architecture": config.architecture, "rope_parameters": config.hf_config.rope_parameters,
            "cases": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preflight-receipt", type=Path)
    args = parser.parse_args()
    provenance = build(args.snapshot, args.output)
    print(json.dumps(provenance, indent=2))
    if args.preflight_receipt:
        result = {"provenance": provenance, "native_processor": preflight(args.output)}
        args.preflight_receipt.parent.mkdir(exist_ok=True, parents=True)
        args.preflight_receipt.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
