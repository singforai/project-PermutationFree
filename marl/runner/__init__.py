"""Runner registry."""
from .smacv2_runner import SMACv2Runner

RUNNER_REGISTRY = {
    "smacv2": SMACv2Runner,
}
