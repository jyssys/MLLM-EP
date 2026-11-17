"""Describe token-axis work at its original request-record boundary."""
import json
import os
from pathlib import Path
from .token_axis import suffix_for


def annotate(original):
    def append(path, value):
        if Path(path).name == 'requests.jsonl':
            cfg = json.loads(Path(os.environ['DENSE_CONFIG']).read_text())
            if cfg.get('token_axis'):
                suffix = suffix_for(cfg, 'q' + str(value['sample_id']))
                value = dict(value, actual_axis='TOKEN', actual_token_suffix=suffix,
                    D_completed_prompt_suffix_tokens=suffix,
                    D_completed_prompt_token_layer_rows=suffix * 36,
                    first_sample_scope='Discard the P prefix sample. Import only its valid prefix KV. D executes the remaining prompt tokens through all36 layers and generates the first user output.')
        return original(path, value)
    return append
