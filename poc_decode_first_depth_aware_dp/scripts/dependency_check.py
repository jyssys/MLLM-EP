"""CPU feedback loop for the official engine's exact dependency versions."""
from poc_decode_first_depth_aware_dp.contract import *
from poc_decode_first_depth_aware_dp.runtime import configure_imports


def main():
    configure_imports()
    import torch
    import transformers
    import flash_attn_interface
    from looped_cdb.models.ouro import OuroConfig
    config=OuroConfig.from_pretrained(MODEL,pad_token_id=None)
    config.pad_token_id=None
    assert config.pad_token_id is None
    assert config.total_ut_steps==4
    print('CDB_DEPENDENCIES_GREEN',torch.__version__,transformers.__version__,flash_attn_interface.__file__,config.pad_token_id)


if __name__=='__main__':main()
