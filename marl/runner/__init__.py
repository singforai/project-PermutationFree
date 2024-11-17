"""Runner registry."""
from .mappo_runner import MAPPORunner
from .mast_runner import MASTRunner

RUNNER_REGISTRY = {
    "mast": MASTRunner,
    "mappo": MAPPORunner,
}
