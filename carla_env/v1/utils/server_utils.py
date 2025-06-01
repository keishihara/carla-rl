from __future__ import annotations

import getpass
import logging
import os
import signal
import socket
import subprocess
import time

import psutil

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


def find_free_port(start_port: int, max_range: int = 1000) -> int:
    """Find an available TCP port greater than or equal to the given port.

    Args:
        start_port (int): The starting port number to search from.
        max_range (int): The maximum range to search for free ports.

    Returns:
        int: An available port number.

    Raises:
        RuntimeError: If no free port is found in the range [port, port+1000).
    """
    for p in range(start_port, start_port + max_range):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", p))
                return p
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start_port}-{start_port + max_range}")


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
        rpc_port: int = 2000,
        streaming_port: int = 0,
        server_start_timeout: float = 60.0,
        server_life_time: float = 1800,
        use_apptainer: bool | str = "auto",
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
            server_start_timeout: Timeout in seconds for startup check
            server_life_time: Maximum server lifetime in seconds
            use_apptainer: If True/auto, launch via Apptainer; otherwise use host script

        """
        self.container_image = container_image
        self.container_script = container_script
        self.host_script = host_script
        self.dummy_file = dummy_file
        self.gpu_index = gpu_index
        self.rpc_port = rpc_port
        self.streaming_port = streaming_port
        self.server_start_timeout = server_start_timeout
        self.server_life_time = server_life_time
        self.use_apptainer = use_apptainer

        self._start_time = 0
        self._process: subprocess.Popen | None = None

    def start(
        self,
        gpu_index: int | None = None,
        rpc_port: int | None = None,
        wait_time_after_start: int = 3,
    ) -> None:
        """Start the CARLA server on the specified GPU.

        Args:
            gpu_index: GPU index to use (overrides constructor value)
            rpc_port: RPC port to use (overrides constructor value)
            wait_time_after_start: Time to wait after starting server

        Raises:
            RuntimeError: If the server does not start or is already running
        """
        if self._process and self._process.poll() is None:
            raise RuntimeError("Carla server is already running")

        if gpu_index is not None:
            self.gpu_index = gpu_index
        if rpc_port is not None:
            self.rpc_port = rpc_port

        self._start_time = time.time()
        cmd = self._build_command()
        self._process = subprocess.Popen(cmd, start_new_session=True)
        time.sleep(wait_time_after_start)

        if not self._is_server_running():
            # kill any partial process if startup failed
            self.stop()
            raise RuntimeError(f"Failed to start Carla server on rpc_port {self.rpc_port}")

        logger.info(f"CARLA server started on port {self.rpc_port} using GPU {self.gpu_index}")

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

        # 2) Wait up to 10 seconds for the process to terminate
        deadline = time.time() + 10
        while time.time() < deadline:
            # If the LISTEN is gone, it's ok
            if not any(
                c.laddr.port == self.rpc_port and c.status == psutil.CONN_LISTEN
                for c in psutil.net_connections(kind="tcp")
            ):
                break
            time.sleep(1)
        else:
            # Timeout: force kill
            for conn in psutil.net_connections(kind="tcp"):
                if conn.laddr.port == self.rpc_port:
                    try:
                        psutil.Process(conn.pid).kill()
                    except Exception:
                        pass

        # 3) Also terminate the Apptainer process group
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

    def restart(self, gpu_index: int | None = None, rpc_port: int | None = None) -> None:
        """Restarts the running Carla server, optionally on a different GPU and/or RPC port."""
        self.stop()

        if gpu_index is not None:
            self.gpu_index = gpu_index
        if rpc_port is not None:
            self.rpc_port = rpc_port

        logger.info(f"Restarting Carla server on GPU {self.gpu_index} and RPC port {self.rpc_port}")
        self.start()

    close = stop

    def _build_command(self) -> list[str]:
        """
        Constructs the command to launch Carla, choosing container or host mode.

        Returns:
            List of command arguments.
        """
        using_apptainer = self._should_use_apptainer()
        script = self.container_script if using_apptainer else self.host_script

        self.streaming_port = 0
        self.traffic_manager_port = find_free_port(self.rpc_port + 6005)

        cmd = [
            script,
            "-RenderOffScreen",
            "-nosound",
            f"-graphicsadapter={self.gpu_index}",
            f"-carla-world-port={self.rpc_port}",
            f"-carla-streaming-port={self.streaming_port}",
            f"-trafficmanager-port={self.traffic_manager_port}",
        ]
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
        start_time = time.time()
        while time.time() - start_time < self.server_start_timeout:
            try:
                with socket.create_connection(("localhost", self.rpc_port), timeout=1):
                    return True
            except (ConnectionRefusedError, socket.timeout):
                time.sleep(3)
        return False


def single_gpu_demo():
    import argparse
    import gc

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--rpc-port", type=int, default=3000)
    parser.add_argument("--streaming-port", type=int, default=0)
    parser.add_argument("--town", type=str, default="Town01")
    args = parser.parse_args()

    manager = CarlaServerManager(
        gpu_index=args.gpu_id,
        rpc_port=args.rpc_port,
        streaming_port=args.streaming_port,
    )

    kill_carla()

    def connect_carla(port):
        import carla

        client = carla.Client("localhost", port)
        client.set_timeout(60.0)
        return client

    def set_sync_mode(client, port, tm=None, sync=True):
        world = client.get_world()
        settings = world.get_settings()
        settings.synchronous_mode = sync
        settings.fixed_delta_seconds = 0.05
        settings.deterministic_ragdolls = True
        world.apply_settings(settings)
        if tm is None:
            tm = client.get_trafficmanager(port + 6000)
        tm.set_synchronous_mode(sync)
        return world, tm

    try:
        manager.start(gpu_index=args.gpu_id)

        client = connect_carla(args.rpc_port)
        world, tm = set_sync_mode(client, args.rpc_port, sync=True)
        world.tick()

        set_sync_mode(client, args.rpc_port, tm, sync=False)
        client = None
        tm.shut_down()
        del client
        del world
        del tm
        gc.collect()

        new_rpc_port = args.rpc_port  #  + 50
        manager.restart(gpu_index=(args.gpu_id + 1) % len(get_gpu_names()), rpc_port=new_rpc_port)

        client = connect_carla(new_rpc_port)
        world, tm = set_sync_mode(client, new_rpc_port, sync=True)
        world.tick()

    except RuntimeError as e:
        print(f"Error: {e}")
    finally:
        manager.stop()


if __name__ == "__main__":
    single_gpu_demo()
