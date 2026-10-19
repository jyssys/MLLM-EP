"""Resolve pinned read-only binary libraries before the isolated torchrun launcher."""
from poc_ouro_parallelism_deep_discovery.runtime import configure_imports
configure_imports()
from torch.distributed.run import main

if __name__=='__main__':main()
