"""Reuse the frozen lossless packer in this campaign namespace only."""
from . import common


def main():
    from ..dense_ar_dynamic_layer_pd import archive_evidence as implementation
    for name in ('ROOT','REPO','resources','digest','write'):
        setattr(implementation,name,getattr(common,name))
    implementation.main()


if __name__=='__main__':main()
