"""Apply the new fixed deadline through the existing owned-session guard."""
from .common import configure

if __name__ == '__main__':
    configure()
    from ..looped_pd_autonomous_headroom.deadline_guard import main
    main()
