from __future__ import annotations

import getpass
import logging
import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass

import carla
import psutil

from carla_env.v2.utils.port import find_random_free_port

logger = logging.getLogger(__name__)

CONTAINER_IMAGE = f"/home/{getpass.getuser()}/workspace/apptainer_images/carla_0.9.15_sandbox2"
CONTAINER_CARLA_SCRIPT = f"/home/{getpass.getuser()}/carla_0_9_15/CarlaUE4.sh"
HOST_CARLA_SCRIPT = f"/home/{getpass.getuser()}/workspace/carla_0_9_15/CarlaUE4.sh"
DUMMY_FILE = f"/home/{getpass.getuser()}/empty/dummy_nvidia"  # for GPU binding inside the container


def kill_carla(port: int | None = None, wait_time: int = 5):
    """Kill CARLA server on the specified port or all servers if no port is specified.

    Args:
        port: Specific port to kill server on, None for all servers
        wait_time: Time to wait after sending kill signals
    """
    if port is None:
        cmd = "ps -ef | grep CarlaUE4 | awk '{print $2}'"
    else:
        cmd = f"pgrep -f 'carla-rpc-port={port}'"

    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    pids, _ = process.communicate()
    pids = pids.strip().split("\n")

    if pids:
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGTERM)
                time.sleep(1)
                os.kill(int(pid), signal.SIGTERM)  # run twice to be sure
            except ProcessLookupError:
                pass

        time.sleep(wait_time)

        # If the process is still running, kill it forcefully
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        pids, _ = process.communicate()
        pids = pids.strip().split("\n")

        for pid in pids:
            if pid:
                try:
                    os.kill(int(pid), signal.SIGKILL)  # SIGKILL to forcefully terminate
                except ProcessLookupError:
                    pass

    time.sleep(wait_time)


def get_gpu_names() -> list[str]:
    """Get list of available GPU names using nvidia-smi.

    Returns:
        List of GPU names or empty list if nvidia-smi fails
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        gpu_names = result.stdout.strip().split("\n")
        return gpu_names
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get GPU names: {e.stderr}")
        return []


@dataclass
class CarlaPorts:
    rpc_port: int
    streaming_port: int
    tm_port: int


class CarlaServerManager:
    """Manager to start and stop a CARLA server, either via Apptainer or directly on the host.

    Features:
    - Automatic port discovery
    - Auto-restart after lifetime expires
    - Container and host execution support
    - GPU binding for containers
    """

    def __init__(
        self,
        container_image: str = CONTAINER_IMAGE,
        container_script: str = CONTAINER_CARLA_SCRIPT,
        host_script: str = HOST_CARLA_SCRIPT,
        dummy_file: str = DUMMY_FILE,
        gpu_index: int = 0,
        rpc_port: int | str = "auto",
        streaming_port: int | str = "auto",
        tm_port: int | str = "auto",
        server_start_timeout: float = 60.0,
        server_life_time: float = 1800,
        use_apptainer: bool | str = "auto",
        nullrhi: bool = False,
    ):
        """Initialize the CarlaServerManager.

        Args:
            container_image: Path to the Apptainer image
            container_script: CARLA startup script inside the container
            host_script: CARLA startup script on the host
            dummy_file: Dummy file for GPU binding in container
            gpu_index: GPU index to use for rendering
            rpc_port: Port for CARLA RPC communication
            streaming_port: Port for CARLA streaming (0 for auto)
            tm_port: Port for CARLA traffic manager (6005 for auto)
            server_start_timeout: Timeout in seconds for startup check
            server_life_time: Maximum server lifetime in seconds
            use_apptainer: If True/auto, launch via Apptainer; otherwise use host script
            nullrhi: If True, use -nullrhi flag added to the command
        """
        self.container_image = container_image
        self.container_script = container_script
        self.host_script = host_script
        self.dummy_file = dummy_file
        self.gpu_index = gpu_index
        self.rpc_port = rpc_port
        self.streaming_port = streaming_port
        self.tm_port = tm_port
        self.server_start_timeout = server_start_timeout
        self.server_life_time = server_life_time
        self.use_apptainer = use_apptainer
        self.nullrhi = nullrhi

        self._start_time = 0
        self._process: subprocess.Popen | None = None

    def start(
        self,
        gpu_index: int | None = None,
        rpc_port: int | None = None,
        nullrhi: bool | None = None,
        wait_time_after_start: int = 10,
        max_retries: int = 3,
    ) -> CarlaPorts:
        """Start the CARLA server on the specified GPU.

        Args:
            gpu_index: GPU index to use (overrides constructor value)
            rpc_port: RPC port to use (overrides constructor value)
            nullrhi: If True, use -nullrhi flag added to the command (overrides constructor value)
            wait_time_after_start: Time to wait after starting server
            max_retries: Maximum number of retries to start the server

        Returns:
            int: The RPC port used by the server

        Raises:
            RuntimeError: If the server does not start or is already running
        """
        if self._process and self._process.poll() is None:
            raise RuntimeError("Carla server is already running")

        if gpu_index is not None:
            self.gpu_index = gpu_index
        if rpc_port is not None:
            self.rpc_port = rpc_port
        if nullrhi is not None:
            self.nullrhi = nullrhi

        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            if self.rpc_port == "auto":
                self.rpc_port = find_random_free_port()
            if self.streaming_port == "auto":
                # self.streaming_port = find_random_free_port()
                self.streaming_port = 0
            if self.tm_port == "auto":
                self.tm_port = find_random_free_port()

            self._start_time = time.monotonic()
            cmd = self._build_command()
            logger.debug(f"[{attempt}/{max_retries}] Start CARLA: {' '.join(cmd)}")
            self._process = subprocess.Popen(cmd, start_new_session=True)
            time.sleep(wait_time_after_start)

            try:
                if not self._is_server_running():
                    raise RuntimeError("RPC port not listening")

                logger.debug("Waiting for simulator to be ready....")
                if not self._wait_for_simulator_ready():
                    raise RuntimeError("Simulator handshake failed")

                logger.info(f"CARLA server (GPU {self.gpu_index}) ready on port {self.rpc_port}")
                return CarlaPorts(
                    rpc_port=self.rpc_port,
                    streaming_port=self.streaming_port,
                    tm_port=self.tm_port,
                )

            except Exception as e:
                last_error = e
                logger.warning(f"[{attempt}/{max_retries}] Startup failed: {e}. Restarting…")
                self.stop()  # clean up before retry

        raise RuntimeError(f"Failed to start CARLA after {max_retries} attempts: {last_error}")

    def stop(self) -> None:
        """
        Stops the Carla server by terminating processes listening on the RPC port
        and killing the Carla process group.
        """

        # 1) Terminate the process listening on the RPC port
        for conn in psutil.net_connections(kind="tcp"):
            if conn.laddr.port == self.rpc_port and conn.status == psutil.CONN_LISTEN:
                try:
                    psutil.Process(conn.pid).terminate()
                except Exception:
                    pass

        # 2) Also terminate the Apptainer process group
        if self._process:
            try:
                os.killpg(self._process.pid, signal.SIGTERM)
                self._process.wait(timeout=5)
            except Exception:
                pass

        self._process = None
        logger.info(f"CARLA server on port {self.rpc_port} stopped")

    def needs_restart(self) -> bool:
        """Check if the server needs to be restarted due to lifetime expiration."""
        return (time.monotonic() - self._start_time) >= self.server_life_time

    def restart(self, gpu_index: int | None = None, rpc_port: int | None = None) -> CarlaPorts:
        """Restarts the running Carla server, optionally on a different GPU and/or RPC port."""
        self.stop()
        return self.start(gpu_index=gpu_index, rpc_port=rpc_port)

    close = stop

    def _build_command(self) -> list[str]:
        """
        Constructs the command to launch Carla, choosing container or host mode.

        Returns:
            List of command arguments.
        """
        using_apptainer = self._should_use_apptainer()
        script = self.container_script if using_apptainer else self.host_script

        cmd = [
            script,
            "-RenderOffScreen",
            "-nosound",
            f"-graphicsadapter={self.gpu_index}",
            f"-carla-rpc-port={self.rpc_port}",
            f"-carla-streaming-port={self.streaming_port}",
            f"-trafficmanager-port={self.tm_port}",
        ]
        if self.nullrhi:
            cmd.append("-nullrhi")
        if using_apptainer:
            cmd = [
                "apptainer",
                "exec",
                "--nv",
                "--writable-tmpfs",
                self.container_image,
                *cmd,
            ]

        return cmd

    def _should_use_apptainer(self) -> bool:
        """Determine if Apptainer should be used based on configuration."""
        if self.use_apptainer == "auto":
            gpu_names = get_gpu_names()
            return any(["h100" in name.lower() for name in gpu_names])

        return self.use_apptainer

    def _is_server_running(self) -> bool:
        """Check if the CARLA server is listening on the RPC port."""
        start_time = time.monotonic()
        while time.monotonic() - start_time < self.server_start_timeout:
            try:
                with socket.create_connection(("localhost", self.rpc_port), timeout=1):
                    return True
            except (ConnectionRefusedError, socket.timeout):
                time.sleep(1)
        return False

    def _wait_for_simulator_ready(self, timeout: float = 120.0) -> bool:
        """Return True when the simulator answers basic RPCs within *timeout*.

        Args:
            timeout: Seconds to keep trying before giving up.

        """
        start_time = time.monotonic()
        deadline = start_time + timeout
        client = carla.Client("localhost", self.rpc_port)
        client.set_timeout(10.0)
        while time.monotonic() < deadline:
            try:
                client.get_server_version()
                client.get_world().get_map()
                return True
            except KeyboardInterrupt:
                raise KeyboardInterrupt("Keyboard interrupt by user") from None
            except Exception:
                elapsed = time.monotonic() - start_time
                remaining = deadline - time.monotonic()
                logger.debug(f"Simulator not ready yet. Elapsed: {elapsed:.0f}s, Remaining: {remaining:.0f}s")
                time.sleep(1.0)
        return False


def wait_for_simulator_ready(client: carla.Client, max_retries: int = 6, retry_delay: float = 2.0) -> bool:
    """Wait for CARLA simulator to be fully ready.

    Args:
        client: CARLA client instance
        max_retries: Maximum number of retry attempts (default: 5)
        retry_delay: Delay between retry attempts in seconds

    Returns:
        True if simulator becomes ready, False if max retries exceeded
    """
    for attempt in range(max_retries):
        try:
            world = client.get_world()
            world.get_map()
            world.get_settings()

            logger.info(f"Simulator is ready after {attempt + 1} attempts!")
            return True

        except KeyboardInterrupt:
            raise KeyboardInterrupt("Keyboard interrupt") from None

        except Exception as e:
            logger.debug(f"Simulator not ready yet (attempt {attempt + 1}/{max_retries}): {e}")

            if attempt < max_retries - 1:  # Don't sleep on last attempt
                time.sleep(retry_delay)

    logger.error(f"Simulator failed to become ready after {max_retries} attempts")
    return False


def single_gpu_demo():
    import argparse
    import gc

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--rpc-port", type=int, default=None)
    parser.add_argument("--streaming-port", type=int, default=None)
    parser.add_argument("--tm-port", type=int, default=None)
    parser.add_argument("--town", type=str, default="Town01")
    args = parser.parse_args()

    args.rpc_port = args.rpc_port or "auto"
    args.streaming_port = args.streaming_port or "auto"
    args.tm_port = args.tm_port or "auto"

    manager = CarlaServerManager(
        gpu_index=args.gpu_id,
        rpc_port=args.rpc_port,
        streaming_port=args.streaming_port,
        tm_port=args.tm_port,
    )

    kill_carla()

    def connect_carla(port: int, max_retries: int = 5, retry_delay: float = 2.0) -> carla.Client:
        for attempt in range(max_retries):
            try:
                client = carla.Client("localhost", port)
                client.set_timeout(60.0)
                client.get_server_version()
                return client
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Failed to connect to CARLA server on port {port} after {max_retries} attempts: {e}"
                    ) from e
                print(f"Failed to connect to CARLA server on port {port} after {attempt + 1} attempts. Retrying... {e}")
                time.sleep(retry_delay)

    def set_sync_mode(client: carla.Client, ports: CarlaPorts, tm=None, sync=True):
        world = client.get_world()
        settings = world.get_settings()
        settings.synchronous_mode = sync
        settings.fixed_delta_seconds = 0.05
        settings.deterministic_ragdolls = True
        world.apply_settings(settings)
        if tm is None:
            tm = client.get_trafficmanager(ports.tm_port)
        tm.set_synchronous_mode(sync)
        return world, tm

    try:
        print("[Starting CARLA server]")

        ports = manager.start(gpu_index=args.gpu_id)
        print(f"CARLA server started on port {ports.rpc_port}, tm_port {ports.tm_port}")

        client = connect_carla(ports.rpc_port)
        print(f"Connected to CARLA server on port {ports.rpc_port}")

        if not wait_for_simulator_ready(client):
            raise RuntimeError("Simulator failed to become ready.")

        world, tm = set_sync_mode(client, ports, sync=True)
        print("Setting sync mode to True")
        world.tick()
        print("Ticked world")

        print("Setting sync mode to False, shutting down traffic manager")
        set_sync_mode(client, ports, tm, sync=False)
        client = None
        tm.shut_down()
        del client
        del world
        del tm
        gc.collect()

        print("\n\n[Restarting CARLA server]")

        ports = manager.restart(gpu_index=(args.gpu_id + 1) % len(get_gpu_names()))
        print(f"CARLA server restarted on port {ports.rpc_port}, tm_port {ports.tm_port}")

        client = connect_carla(ports.rpc_port)
        print(f"Connected to CARLA server on port {ports.rpc_port}")

        if not wait_for_simulator_ready(client):
            raise RuntimeError("Simulator failed to become ready.")

        world, tm = set_sync_mode(client, ports, sync=True)
        world.tick()

    except RuntimeError as e:
        print(f"Error: {e}")
        print(type(e))
    finally:
        print("[Stopping CARLA server]")
        manager.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
    )
    single_gpu_demo()
