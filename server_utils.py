from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import time
from typing import List, Optional

import psutil

logger = logging.getLogger(__name__)

# apptainer exec \
#   --nv \
#   --writable-tmpfs \
#   --bind /home/keishi_ishihara/empty/dummy_nvidia:/dev/nvidia1 \
#   /home/keishi_ishihara/workspace/apptainer_images/carla_0.9.15_sandbox2 \
#   /home/carla_0_9_15/CarlaUE4.sh -nosound -RenderOffScreen

CONTAINER_IMAGE = "/home/keishi_ishihara/workspace/apptainer_images/carla_0.9.15_sandbox2"
CONTAINER_CARLA_SCRIPT = "/home/carla_0_9_15/CarlaUE4.sh"
HOST_CARLA_SCRIPT = "/home/ubuntu/workspace/carla_0_9_15/CarlaUE4.sh"
DUMMY_FILE = "/home/keishi_ishihara/empty/dummy_nvidia"  # for GPU binding inside the container


def kill_carla(port: Optional[int] = None, wait_time: int = 5):
    """Kill Carla server on the specified port or all servers if no port is specified."""

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


class CarlaServerManager:
    """
    Manager to start and stop a Carla server, either via Apptainer or directly on the host.
    """

    def __init__(
        self,
        container_image: str = CONTAINER_IMAGE,
        container_script: str = CONTAINER_CARLA_SCRIPT,
        host_script: str = HOST_CARLA_SCRIPT,
        dummy_file: str = DUMMY_FILE,
        rpc_port: int = 2000,
        streaming_port: int = 0,
        server_start_timeout: float = 60.0,
        server_life_time_sec: float = 1800,
        use_apptainer: bool = False,
    ):
        """
        Initializes the CarlaServerManager.

        Args:
            container_image: Path to the Apptainer image.
            container_script: Carla startup script inside the container.
            host_script: Carla startup script on the host.
            dummy_file: Dummy file for GPU binding in container.
            rpc_port: Port for Carla RPC.
            streaming_port: Port for Carla streaming.
            server_start_timeout: Timeout in seconds for startup check.
            server_life_time_sec: Maximum server lifetime in seconds.
            use_apptainer: If True, launch via Apptainer; otherwise use host script.
        """
        self.container_image = container_image
        self.container_script = container_script
        self.host_script = host_script
        self.dummy_file = dummy_file
        self.rpc_port = rpc_port
        self.streaming_port = streaming_port
        self.server_start_timeout = server_start_timeout
        self.server_life_time_sec = server_life_time_sec
        self.use_apptainer = use_apptainer

        self._gpu_index: Optional[int] = None
        self._process: Optional[subprocess.Popen] = None

    def start(self, gpu_index: int = 0, wait_time_after_start: int = 3) -> None:
        """Starts the Carla server on the specified GPU.

        Raises:
            RuntimeError: If the server does not start or is already running.
        """
        if self._process and self._process.poll() is None:
            raise RuntimeError("Carla server is already running")

        cmd = self._build_command(gpu_index)
        self._process = subprocess.Popen(cmd, start_new_session=True)
        self._gpu_index = gpu_index
        time.sleep(wait_time_after_start)

        if not self._is_server_running():
            # kill any partial process if startup failed
            self.stop()
            raise RuntimeError(f"Failed to start Carla server on rpc_port {self.rpc_port}")

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

    def restart(self, gpu_index: int = None, new_rpc_port: int = None) -> None:
        """Restarts the running Carla server, optionally on a different GPU and/or RPC port."""
        self.stop()
        if new_rpc_port is not None:
            self.rpc_port = new_rpc_port
        target_gpu = gpu_index if gpu_index is not None else (self._gpu_index or 0)

        print(f"Restarting Carla server on GPU {target_gpu} and RPC port {self.rpc_port}")
        self.start(gpu_index=target_gpu)

    def _build_command(self, gpu_index: int) -> List[str]:
        """
        Constructs the command to launch Carla, choosing container or host mode.

        Args:
            gpu_index: Index of GPU to assign.

        Returns:
            List of command arguments.
        """
        script = self.container_script if self.use_apptainer else self.host_script

        cmd = [
            script,
            "-RenderOffScreen",
            "-nosound",
            f"-graphicsadapter={gpu_index}",
            f"-carla-streaming-port={self.streaming_port}",
            f"-carla-world-port={self.rpc_port}",
            f"-trafficmanager-port={self.rpc_port + 6000}",
        ]
        if self.use_apptainer:
            cmd = [
                "apptainer",
                "exec",
                "--nv",
                "--writable-tmpfs",
                self.container_image,
                *cmd,
            ]

        return cmd

    def _is_server_running(self) -> bool:
        """
        Checks if the Carla server is listening on the RPC port.

        Returns:
            True if server is up before timeout.
        """
        start_time = time.time()
        while time.time() - start_time < self.server_start_timeout:
            try:
                with socket.create_connection(("localhost", self.rpc_port), timeout=1):
                    return True
            except (ConnectionRefusedError, socket.timeout):
                time.sleep(5)
        return False


def get_gpu_names() -> List[str]:
    try:
        # Run the nvidia-smi command
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Split the output into lines and return as a list
        gpu_names = result.stdout.strip().split("\n")
        return gpu_names
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e.stderr}")
        return []


def single_gpu_demo():
    import argparse
    import gc

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--rpc-port", type=int, default=3000)
    parser.add_argument("--streaming-port", type=int, default=0)
    parser.add_argument("--town", type=str, default="Town01")
    args = parser.parse_args()

    gpu_names = get_gpu_names()
    use_apptainer = any(["h100" in name.lower() for name in gpu_names])
    print(f"Using Apptainer: {use_apptainer}")

    rpc_port = args.rpc_port
    streaming_port = args.streaming_port
    manager = CarlaServerManager(rpc_port=rpc_port, streaming_port=streaming_port, use_apptainer=use_apptainer)

    kill_carla()

    def connect_carla(port):
        import carla

        client = carla.Client("localhost", port)
        client.set_timeout(20.0)
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

        client = connect_carla(rpc_port)
        world, tm = set_sync_mode(client, rpc_port, sync=True)
        world.tick()

        set_sync_mode(client, rpc_port, tm, sync=False)
        client = None
        tm.shut_down()  # This is very important  !!!
        del client
        del world
        del tm
        gc.collect()

        new_rpc_port = args.rpc_port  #  + 50
        manager.restart(new_rpc_port=new_rpc_port)

        client = connect_carla(new_rpc_port)
        world, tm = set_sync_mode(client, new_rpc_port, sync=True)
        world.tick()

    except RuntimeError as e:
        print(f"Error: {e}")
    finally:
        manager.stop()


def multi_gpu_demo():
    import argparse
    import gc

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-ids", type=int, default=[0, 1])
    parser.add_argument("--rpc-port", type=int, default=3000)
    parser.add_argument("--streaming-port", type=int, default=0)
    parser.add_argument("--town", type=str, default="Town01")
    args = parser.parse_args()

    gpu_names = get_gpu_names()
    use_apptainer = any(["h100" in name.lower() for name in gpu_names])
    print(f"Using Apptainer: {use_apptainer}")

    rpc_port = args.rpc_port
    streaming_port = args.streaming_port
    port_offset = 1000
    managers = [
        CarlaServerManager(
            rpc_port=rpc_port + port_offset * i, streaming_port=streaming_port, use_apptainer=use_apptainer
        )
        for i in args.gpu_ids
    ]

    kill_carla()

    def connect_carla(port):
        import carla

        client = carla.Client("localhost", port)
        client.set_timeout(20.0)
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
        for i, manager in enumerate(managers):
            manager.start(gpu_index=args.gpu_ids[i])

        clients = [connect_carla(rpc_port + port_offset * i) for i in args.gpu_ids]
        worlds_tms = [set_sync_mode(client, rpc_port + port_offset * i, sync=True) for i, client in enumerate(clients)]
        for world, tm in worlds_tms:
            world.tick()

        # set_sync_mode(client, rpc_port, tm, sync=False)

        for client in clients:
            client = None
            del client

        for world, tm in worlds_tms:
            tm.shut_down()  # This is very important  !!!
            del world
            del tm
            gc.collect()

        new_rpc_port = args.rpc_port
        for i, manager in enumerate(managers):
            manager.restart(new_rpc_port=new_rpc_port + port_offset * i)

        clients = [connect_carla(new_rpc_port + port_offset * i) for i in args.gpu_ids]
        worlds_tms = [
            set_sync_mode(client, new_rpc_port + port_offset * i, sync=True) for i, client in enumerate(clients)
        ]
        for world, tm in worlds_tms:
            world.tick()

    except RuntimeError as e:
        print(f"Error: {e}")
    finally:
        for manager in managers:
            manager.stop()


if __name__ == "__main__":
    multi_gpu_demo()
