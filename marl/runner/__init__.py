"""Runner registry."""
from .smacv2_runner import SMACv2Runner

RUNNER_REGISTRY = {
    "mast": SMACv2Runner,
    "mappo": SMACv2Runner,
}
