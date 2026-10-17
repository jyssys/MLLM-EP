"""Reuse the publication scanner with the current owned artifact root."""
from poc_recurrent_asymmetry.contract import ROOT,PROJECT
from poc_async_recurrent_pp.scripts import audit_publication as reused


if __name__=='__main__':
    reused.ROOT,reused.PROJECT=ROOT,PROJECT
    reused.main()
