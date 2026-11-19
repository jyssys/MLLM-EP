"""Bind the proven all-eight burner closeout to this evidence namespace."""
from . import common


def main():
    from ..dense_ar_layer_token_pd import resource_closeout as proven
    for key in ('ROOT','QUIET_LOCK','resources','preserve','write','digest'):
        setattr(proven,key,getattr(common,key))
    proven.main()


if __name__=='__main__':
    main()
