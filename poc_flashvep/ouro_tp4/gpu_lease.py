"""Reuse the checked burner lifecycle without writing the old result directory."""
from .contract import ROOT
from poc_flashvep.ouro_request_pp import gpu_lease


if __name__=='__main__':
    gpu_lease.ROOT=ROOT
    gpu_lease.main()
