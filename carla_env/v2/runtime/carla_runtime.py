# carla_runtime.py
"""High-level runtime that coordinates CarlaServerManager and CarlaClientManager."""

from __future__ import annotations

import contextlib
import logging

import carla

from carla_env.v2.runtime.carla_client import CarlaClientManager
from carla_env.v2.runtime.carla_server import CarlaPorts, CarlaServerManager

logger = logging.getLogger(__name__)


class CarlaRuntime:
    """Lifecycle manager for a CARLA server–client pair.

    Responsibilities
    ---------------
    * Boot and shut down the server (`CarlaServerManager`)
    * Initialise the client (`CarlaClientManager`)
    * Perform health checks and restart both sides when necessary
    """

    def __init__(
        self,
        map_name: str = "Town01",
        *,
        gpu_id: int = 0,
        no_rendering_mode: bool = False,
        server_cfg: dict | None = None,
        client_cfg: dict | None = None,
        max_start_retries: int = 3,
    ) -> None:
        """Create a new runtime.

        Args:
            map_name: World to load after connection.
            gpu_id: GPU index used by the server.
            no_rendering_mode: Pass-through to `CarlaClientManager.init`.
            server_cfg: Extra kwargs forwarded to `CarlaServerManager`.
            client_cfg: Extra kwargs forwarded to `CarlaClientManager`.
            max_start_retries: Max attempts to (re)start if the client fails.
        """
        self._map_name = map_name
        self._no_rendering_mode = no_rendering_mode
        self._max_start_retries = max_start_retries

        self._server = CarlaServerManager(gpu_index=gpu_id, **(server_cfg or {}))
        self._client = CarlaClientManager(**(client_cfg or {}))

        self._ports: CarlaPorts | None = None

    @property
    def client(self) -> carla.Client:
        if not self._client.is_initialized:
            raise RuntimeError("Client is not initialized")
        return self._client.client

    @property
    def world(self) -> carla.World:
        if not self._client.is_initialized:
            raise RuntimeError("Client is not initialized")
        return self._client.world

    @property
    def tm(self) -> carla.TrafficManager:
        if not self._client.is_initialized:
            raise RuntimeError("Client is not initialized")
        return self._client.tm

    @property
    def no_rendering_mode(self) -> bool:
        return self._no_rendering_mode

    def set_gpu_id(self, gpu_id: int) -> None:
        if gpu_id == self._server.gpu_index:
            return
        self._server.gpu_index = gpu_id
        logger.info(f"Setting gpu_id to {gpu_id}. This will be applied to the next server start.")

    def set_nullrhi(self, value: bool) -> None:
        if value == self._server.nullrhi:
            return
        logger.info(f"Setting nullrhi to {value}. This will be applied to the next server start.")
        self._server.nullrhi = value

    def set_no_rendering_mode(self, value: bool, *, immediate: bool = False) -> None:
        if value == self._no_rendering_mode:
            return

        self._no_rendering_mode = value
        if immediate:
            self._client.set_no_rendering_mode(value)

    def start(self) -> CarlaPorts:
        """Boot the server and connect the client. Retries on failure."""
        last_err: Exception | None = None
        for attempt in range(1, self._max_start_retries + 1):
            try:
                self._ports = self._server.start()
                self._client.init(
                    map_name=self._map_name,
                    rpc_port=self._ports.rpc_port,
                    tm_port=self._ports.tm_port,
                    no_rendering_mode=self._no_rendering_mode,
                )
                logger.info(f"CarlaRuntime ready (attempt {attempt})")
                return self._ports
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                logger.warning(f"Startup attempt {attempt} failed: {exc}. Restarting…")
                self._server.stop()
        raise RuntimeError(f"Failed to launch CARLA runtime: {last_err}")

    def ensure_healthy(self) -> bool:
        """Restart server/client if unhealthy.

        Returns:
            bool: ``True`` if a restart occurred, else ``False``.
        """
        restarted = False
        if self._server.needs_restart() or not self._client.is_connection_healthy():
            restarted = True
            logger.info("Restarting CARLA runtime (health check failed)")
            self._client.close()
            self._ports = self._server.restart()
            self._client.init(
                map_name=self._map_name,
                rpc_port=self._ports.rpc_port,
                tm_port=self._ports.tm_port,
                no_rendering_mode=self._no_rendering_mode,
            )
        return restarted

    def is_running(self) -> bool:
        return self._client.is_connection_healthy()

    def close(self) -> None:
        """Gracefully shut down client and server."""
        with contextlib.suppress(Exception):
            self._client.close()
        with contextlib.suppress(Exception):
            self._server.close()

    def __enter__(self) -> CarlaRuntime:  # noqa: D401
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:  # noqa: D401
        self.close()
        # propagate exceptions
        return False
