from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class ProviderSnapshot:
    as_of: str
    asset_ids: tuple[str, ...]
    horizon: str
    source: str
    version: str
    payload: Any


class PredictionProvider(Protocol):
    def snapshot(self, *, as_of: str, asset_ids: tuple[str, ...], horizon: str) -> ProviderSnapshot: ...


class RiskModelProvider(Protocol):
    def snapshot(self, *, as_of: str, asset_ids: tuple[str, ...], horizon: str) -> ProviderSnapshot: ...


class BenchmarkWeightProvider(Protocol):
    def snapshot(self, *, as_of: str, asset_ids: tuple[str, ...], horizon: str) -> ProviderSnapshot: ...


def required_inputs(capabilities: Mapping[str, Any], model: Mapping[str, Any]) -> tuple[str, ...]:
    identity = (model.get("type"), model.get("version"))
    for item in capabilities.get("models", []):
        if item.get("type") == identity[0] and identity[1] in item.get("versions", []):
            values = item.get("required_inputs", [])
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise ValueError("optimizer capability required_inputs is malformed")
            return tuple(values)
    raise ValueError(f"optimizer model capability is missing: {identity[0]}/{identity[1]}")


def resolve_snapshots(
    dependencies: tuple[str, ...],
    providers: Mapping[str, Any],
    *,
    as_of: str,
    asset_ids: tuple[str, ...],
    horizon: str,
) -> dict[str, ProviderSnapshot]:
    snapshots: dict[str, ProviderSnapshot] = {}
    for dependency in dependencies:
        provider = providers.get(dependency)
        if provider is None:
            raise ValueError(f"missing optimizer data provider: {dependency}")
        snapshot = provider.snapshot(as_of=as_of, asset_ids=asset_ids, horizon=horizon)
        if snapshot.as_of != as_of or snapshot.asset_ids != asset_ids or snapshot.horizon != horizon:
            raise ValueError(f"provider snapshot metadata mismatch: {dependency}")
        snapshots[dependency] = snapshot
    return snapshots
