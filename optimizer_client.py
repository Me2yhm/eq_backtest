"""Optimizer backend selection for EQ Backtest.

The implementation lives in the installable optimizer package.  This module
keeps the existing EQ Backtest import surface small and stable.
"""

from optimizer import InProcessOptimizerClient, OptimizerClientError, OptimizerLease
from optimizer.transport.shm import ShmOptimizerClient


OptimizerClient = ShmOptimizerClient


def create_optimizer_client(config: dict):
    mode = config["mode"]
    if mode == "inprocess":
        return InProcessOptimizerClient(
            schema_version=config["schema_version"],
            timeout_ms=config["timeout_ms"],
            max_assets=config["max_assets"],
        )
    if mode == "shared_memory":
        return ShmOptimizerClient(
            config["socket_path"],
            protocol_version=config["protocol_version"],
            schema_version=config["schema_version"],
            timeout_ms=config["timeout_ms"],
            request_timeout_ms=config["request_timeout_ms"],
            max_control_bytes=config["max_control_bytes"],
            max_shared_bytes=config["max_shared_bytes"],
        )
    raise ValueError(f"unsupported optimizer mode: {mode!r}")


__all__ = [
    "InProcessOptimizerClient",
    "OptimizerClient",
    "OptimizerClientError",
    "OptimizerLease",
    "ShmOptimizerClient",
    "create_optimizer_client",
]
