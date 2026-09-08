"""shm/1 client for the local optimizer service."""

from __future__ import annotations

import json
import math
import mmap
import os
import socket
import sys
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class OptimizerClientError(RuntimeError):
    def __init__(self, message: str, response: dict | None = None) -> None:
        super().__init__(message)
        self.response = response


def _send_packet(sock: socket.socket, payload: dict[str, Any], max_control_bytes: int | None = None) -> None:
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if max_control_bytes is not None and len(encoded) > max_control_bytes:
        raise OptimizerClientError("optimizer control request exceeded configured boundary")
    if sock.send(encoded) != len(encoded):
        raise OptimizerClientError("optimizer control send was truncated")


def _recv_packet(sock: socket.socket, max_control_bytes: int) -> tuple[dict, tuple[int, ...]]:
    raw, ancdata, flags, _ = sock.recvmsg(max_control_bytes + 1, socket.CMSG_SPACE(16))
    fds: list[int] = []
    for level, kind, data in ancdata:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            values = array("i")
            values.frombytes(data[: len(data) - len(data) % values.itemsize])
            fds.extend(values)
    if not raw:
        for fd in fds:
            os.close(fd)
        raise OptimizerClientError("optimizer closed the control connection")
    if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or len(raw) > max_control_bytes:
        for fd in fds:
            os.close(fd)
        raise OptimizerClientError("optimizer control frame exceeded configured boundary")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        for fd in fds:
            os.close(fd)
        raise OptimizerClientError("optimizer returned invalid JSON metadata") from exc
    if not isinstance(payload, dict):
        for fd in fds:
            os.close(fd)
        raise OptimizerClientError("optimizer control response must be an object")
    return payload, tuple(fds)


@dataclass(slots=True)
class OptimizerLease:
    client: "OptimizerClient"
    descriptor: dict
    mapping: mmap.mmap
    weights: np.ndarray | None
    response: dict
    received_fd: int
    mapping_validation_ms: float
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.weights = None
        try:
            self.mapping.close()
        except BufferError as exc:
            raise OptimizerClientError("optimizer output is still referenced at release") from exc
        os.close(self.received_fd)
        self.received_fd = -1
        self.released = True
        self.client._consume_and_release(self.descriptor)


class OptimizerClient:
    def __init__(
        self,
        socket_path: str | Path,
        *,
        protocol_version: str = "shm/1",
        schema_version: str = "1.1",
        timeout_ms: int = 5000,
        request_timeout_ms: int = 6000,
        max_control_bytes: int = 1_048_576,
        max_shared_bytes: int = 268_435_456,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.protocol_version = protocol_version
        self.schema_version = schema_version
        self.timeout_ms = timeout_ms
        self.request_timeout_ms = request_timeout_ms
        self.max_control_bytes = max_control_bytes
        self.max_shared_bytes = max_shared_bytes
        self.socket: socket.socket | None = None
        self.session_id = ""
        self.service_epoch = ""
        self.capabilities: dict = {}

    @property
    def envelope(self) -> dict:
        return {
            "protocol_version": self.protocol_version,
            "service_epoch": self.service_epoch,
            "session_id": self.session_id,
        }

    def connect(self) -> None:
        if self.socket is not None:
            return
        if (
            self.protocol_version != "shm/1"
            or not sys.platform.startswith("linux")
            or sys.byteorder != "little"
            or not hasattr(socket, "SCM_RIGHTS")
        ):
            raise OptimizerClientError("shm/1 requires little-endian Linux AF_UNIX and SCM_RIGHTS")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        sock.settimeout(self.request_timeout_ms / 1000.0)
        try:
            sock.connect(str(self.socket_path))
            self.socket = sock
            _send_packet(sock, {"type": "HELLO", "protocol_version": self.protocol_version}, self.max_control_bytes)
            hello, fds = _recv_packet(sock, self.max_control_bytes)
            self._reject_fds(fds)
            if hello.get("type") != "HELLO_OK" or hello.get("protocol_version") != self.protocol_version:
                raise OptimizerClientError("optimizer handshake failed", hello)
            self.session_id = hello.get("session_id", "")
            self.service_epoch = hello.get("service_epoch", "")
            if not self.session_id or not self.service_epoch:
                raise OptimizerClientError("optimizer handshake omitted session identity", hello)
            health = self._round_trip({"type": "HEALTH", **self.envelope})
            if health.get("type") != "HEALTH_RESULT" or health.get("status") != "ok":
                raise OptimizerClientError("optimizer health check failed", health)
            caps = self._round_trip({"type": "CAPABILITIES", **self.envelope})
            self._validate_identity(caps)
            self.capabilities = caps.get("capabilities", {})
            self._validate_capabilities()
        except Exception:
            sock.close()
            self.socket = None
            raise

    def _reject_fds(self, fds: tuple[int, ...]) -> None:
        if fds:
            for fd in fds:
                os.close(fd)
            raise OptimizerClientError("unexpected file descriptor in control response")

    def _round_trip(self, payload: dict) -> dict:
        if self.socket is None:
            raise OptimizerClientError("optimizer client is not connected")
        try:
            _send_packet(self.socket, payload, self.max_control_bytes)
            response, fds = _recv_packet(self.socket, self.max_control_bytes)
        except socket.timeout as exc:
            self.close()
            raise OptimizerClientError("optimizer request timed out; session was abandoned") from exc
        self._reject_fds(fds)
        return response

    def _validate_identity(self, response: dict) -> None:
        if response.get("protocol_version") != self.protocol_version:
            raise OptimizerClientError("optimizer protocol version mismatch", response)
        if response.get("service_epoch") != self.service_epoch or response.get("session_id") != self.session_id:
            raise OptimizerClientError("optimizer session identity mismatch", response)

    def _validate_capabilities(self) -> None:
        caps = self.capabilities
        models = {
            (item.get("type"), version) for item in caps.get("models", []) for version in item.get("versions", [])
        }
        transport = caps.get("transport", {})
        if caps.get("schema_version") != self.schema_version or ("equal_weight", "1") not in models:
            raise OptimizerClientError("optimizer does not support the configured schema/model", caps)
        expected = {
            "protocol_version": "shm/1",
            "control": "unix_domain_socket",
            "backend": "memfd",
            "numeric_dtype": "<f8",
            "order": "C",
        }
        if any(transport.get(key) != value for key, value in expected.items()):
            raise OptimizerClientError("optimizer transport capabilities do not match configuration", caps)

    def validate_capacity(self, n_assets: int) -> None:
        limits = self.capabilities.get("limits", {})
        service_assets = limits.get("max_assets")
        service_bytes = limits.get("max_shared_bytes_per_session")
        if not isinstance(service_assets, int) or n_assets > service_assets:
            raise OptimizerClientError(f"optimizer service cannot accept {n_assets} assets", self.capabilities)
        required_bytes = n_assets * 8
        if not isinstance(service_bytes, int) or required_bytes > min(service_bytes, self.max_shared_bytes):
            raise OptimizerClientError(
                "optimizer shared-memory capacity is below configured portfolio size", self.capabilities
            )

    def optimize(self, request: dict) -> tuple[OptimizerLease, float]:
        if self.socket is None:
            raise OptimizerClientError("optimizer client is not connected")
        started = time.perf_counter_ns()
        try:
            _send_packet(self.socket, {"type": "OPTIMIZE", "request": request, **self.envelope}, self.max_control_bytes)
            response, fds = _recv_packet(self.socket, self.max_control_bytes)
        except socket.timeout as exc:
            self.close()
            raise OptimizerClientError("optimizer request timed out; no fallback was used") from exc
        wait_ms = (time.perf_counter_ns() - started) / 1_000_000
        try:
            self._validate_identity(response)
        except Exception:
            for fd in fds:
                os.close(fd)
            raise
        if response.get("request_id") != request.get("request_id"):
            self._reject_fds(fds)
            raise OptimizerClientError("optimizer request_id mismatch", response)
        if response.get("schema_version") != self.schema_version or response.get("status") != "feasible":
            self._reject_fds(fds)
            raise OptimizerClientError("optimizer rejected the request", response)
        if len(fds) != 1:
            self._reject_fds(fds)
            raise OptimizerClientError("optimizer success response must carry exactly one output handle", response)
        expected_response_fields = {
            "type",
            "protocol_version",
            "service_epoch",
            "session_id",
            "schema_version",
            "request_id",
            "status",
            "warnings",
            "errors",
            "solution",
            "diagnostics",
        }
        if (
            set(response) != expected_response_fields
            or response.get("type") != "OPTIMIZE_RESULT"
            or response.get("warnings") != []
            or response.get("errors") != []
        ):
            os.close(fds[0])
            raise OptimizerClientError("optimizer success response violates the control contract", response)
        solution = response.get("solution", {})
        asset_ids = request["universe"]["asset_ids"]
        if set(solution) != {"asset_ids", "weights"} or solution.get("asset_ids") != asset_ids:
            os.close(fds[0])
            raise OptimizerClientError("optimizer asset ordering mismatch", response)
        descriptor = solution.get("weights")
        expected = {
            "generation": 1,
            "offset": 0,
            "nbytes": len(asset_ids) * 8,
            "shape": [len(asset_ids)],
            "dtype": "<f8",
            "order": "C",
            "role": "output",
        }
        if (
            not isinstance(descriptor, dict)
            or set(descriptor) != {"buffer_id", *expected}
            or not isinstance(descriptor.get("buffer_id"), str)
            or any(descriptor.get(key) != value for key, value in expected.items())
        ):
            os.close(fds[0])
            raise OptimizerClientError("optimizer returned an invalid output descriptor", response)
        diagnostics = response.get("diagnostics")
        model = request.get("model", {})
        if (
            not isinstance(diagnostics, dict)
            or diagnostics.get("model_type") != model.get("type")
            or diagnostics.get("model_version") != model.get("version")
        ):
            os.close(fds[0])
            raise OptimizerClientError("optimizer response model identity mismatch", response)
        mapping_started = time.perf_counter_ns()
        try:
            mapping = mmap.mmap(fds[0], descriptor["nbytes"], access=mmap.ACCESS_READ)
            weights = np.ndarray((len(asset_ids),), dtype="<f8", buffer=mapping)
            weights.setflags(write=False)
            budget = float(request["portfolio_policy"]["target_budget"])
            if not np.isfinite(weights).all() or np.any(weights < 0.0) or np.any(weights > 1.0):
                raise OptimizerClientError("optimizer returned non-finite or out-of-range weights", response)
            if not math.isclose(float(weights.sum()), budget, rel_tol=0.0, abs_tol=1e-10):
                raise OptimizerClientError("optimizer weights violate target_budget", response)
        except Exception:
            if "mapping" in locals():
                if "weights" in locals():
                    del weights
                mapping.close()
            os.close(fds[0])
            raise
        mapping_validation_ms = (time.perf_counter_ns() - mapping_started) / 1_000_000
        return OptimizerLease(self, descriptor, mapping, weights, response, fds[0], mapping_validation_ms), wait_ms

    def _consume_and_release(self, descriptor: dict) -> None:
        consumed = self._round_trip({"type": "CONSUMED", "buffer": descriptor, **self.envelope})
        self._validate_identity(consumed)
        if consumed.get("type") != "CONSUMED_OK":
            raise OptimizerClientError("optimizer rejected output consumption", consumed)
        released = self._round_trip({
            "type": "RELEASE",
            "buffer_id": descriptor["buffer_id"],
            "generation": descriptor["generation"],
            **self.envelope,
        })
        self._validate_identity(released)
        if released.get("type") != "RELEASE_OK":
            raise OptimizerClientError("optimizer rejected output release", released)

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def __enter__(self) -> "OptimizerClient":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
