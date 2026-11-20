"""Reuse the qualified BI0 worker without changing model or transport math."""
from .common import activate
activate()
from ..dense_ar_bi0_shortanswer_pd.standard_worker import main

if __name__=='__main__':
    main()
