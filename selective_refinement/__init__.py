"""Dense-emulation tools for selective masked-token refinement research."""

from .policy import PolicyConfig, choose_active_set

__all__ = ["PolicyConfig", "choose_active_set"]
