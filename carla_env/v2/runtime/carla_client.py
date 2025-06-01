from __future__ import annotations

import logging
import time

import carla

logger = logging.getLogger(__name__)


class CarlaClientManager:
    """
    - Auto search free ports for rpc-port, streaming-port, and tm port
    - Retry connecting to a server
    """

    def __init__(
        self,
        *,
        host: str = "localhost",
        rpc_port: int = 2000,
        tm_port: int = 6005,
        seed: int = 2025,
        timeout: float = 120.0,
    ):
        self._host = host
        self._rpc_port = rpc_port
        self._tm_port = tm_port
        self._seed = seed
        self._timeout = timeout

        self.client: carla.Client | None = None
        self.world: carla.World | None = None
        self.tm: carla.TrafficManager | None = None

        self._client_initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._client_initialized

    def init_legacy(
        self,
        map_name: str,
        *,
        host: str | None = None,
        rpc_port: int | None = None,
        tm_port: int | None = None,
        seed: int | None = None,
        no_rendering_mode: bool = False,
    ) -> None:
        if host is not None:
            self._host = host
        if rpc_port is not None:
            self._rpc_port = rpc_port
        if tm_port is not None:
            self._tm_port = tm_port
        if seed is not None:
            self._seed = seed

        client, world, tm = None, None, None
        server_retry_count = 0
        max_retries_before_restart = 3
        total_retry = 0
        max_total_retries = 12

        while tm is None:
            if total_retry >= max_total_retries:
                raise RuntimeError(f"Failed to connect to Carla server after max retries: {max_total_retries}")

            try:
                client = carla.Client(self._host, self._rpc_port)
                client.set_timeout(self._timeout)
                world = client.load_world(map_name)
                tm = client.get_trafficmanager(self._tm_port)
                server_retry_count = 0  # reset on success
            except RuntimeError as re:
                total_retry += 1
                server_retry_count += 1
                if "timeout" not in str(re) and "time-out" not in str(re):
                    logger.info(f"Could not connect to Carla server because: {re}")
                client = None
                world = None
                tm = None  # back to retry

                if server_retry_count >= max_retries_before_restart:
                    logger.info(
                        f"Restarting Carla server due to repeated connection errors: "
                        f"{server_retry_count}/{max_retries_before_restart}"
                    )
                    # Restart the Carla server after repeated connection errors
                    self._server_manager.restart()
                    server_retry_count = 0
                logger.info(f"Waiting for 5 seconds before retrying... {total_retry}/{max_total_retries}")
                time.sleep(5)

        self.client = client
        self.world = world
        self.tm = tm
        self._client_initialized = True

        self.set_sync_mode(True)
        self.set_no_rendering_mode(no_rendering_mode)

    def init(
        self,
        map_name: str,
        *,
        host: str | None = None,
        rpc_port: int | None = None,
        tm_port: int | None = None,
        seed: int | None = None,
        no_rendering_mode: bool = False,
        max_retries: int = 6,
        retry_delay: float = 5.0,
    ) -> None:
        if host is not None:
            self._host = host
        if rpc_port is not None:
            self._rpc_port = rpc_port
        if tm_port is not None:
            self._tm_port = tm_port
        if seed is not None:
            self._seed = seed

        for attempt in range(1, max_retries + 1):
            try:
                client = carla.Client(self._host, self._rpc_port)
                client.set_timeout(self._timeout)
                world = client.load_world(map_name)
                tm = client.get_trafficmanager(self._tm_port)
                break
            except RuntimeError as re:
                logger.info(f"[{attempt}/{max_retries}] Client connection failed: {re}")
                if attempt == max_retries:
                    raise RuntimeError(f"Failed to connect after {max_retries} attempts") from re
                time.sleep(retry_delay)
        else:  # pragma: no cover
            raise RuntimeError("Unreachable")  # for type checkers

        self.client = client
        self.world = world
        self.tm = tm
        self._client_initialized = True

        self.set_sync_mode(True)
        self.set_no_rendering_mode(no_rendering_mode)

    def set_sync_mode(self, sync: bool) -> None:
        if not self._client_initialized:
            raise RuntimeError("Client not initialized. Please call init_client() first.")

        settings = self.world.get_settings()
        settings.synchronous_mode = sync
        settings.fixed_delta_seconds = 0.05
        settings.deterministic_ragdolls = True
        self.world.apply_settings(settings)
        self.tm.set_synchronous_mode(sync)
        if self._seed is not None:
            self.tm.set_random_device_seed(self._seed)

    def set_no_rendering_mode(self, no_rendering: bool) -> None:
        if not self._client_initialized:
            raise RuntimeError("Client not initialized. Please call init_client() first.")

        settings = self.world.get_settings()
        settings.no_rendering_mode = no_rendering
        self.world.apply_settings(settings)

    def close(self) -> None:
        if not self._client_initialized:
            logger.debug("Client not initialized. Skipping close.")
            return

        try:
            self.set_sync_mode(False)
            self.tm.shut_down()
        except Exception as e:
            logger.debug(f"Client close warning: {e}")
        finally:
            self.client = None
            self.world = None
            self.tm = None
            self._client_initialized = False

    def is_connection_healthy(self) -> bool:
        if self.client is None:
            return False
        try:
            self.client.get_server_version()
            return True
        except Exception:
            return False
