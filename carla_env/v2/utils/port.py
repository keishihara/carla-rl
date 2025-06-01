import random
import socket
from contextlib import closing
from typing import Final

MIN_EPHEMERAL_PORT: Final[int] = 49_152
MAX_EPHEMERAL_PORT: Final[int] = 65_535


def find_random_free_port(
    min_port: int = MIN_EPHEMERAL_PORT,
    max_port: int = MAX_EPHEMERAL_PORT,
    max_attempts: int = 50,
) -> int:
    """Return an unused TCP port chosen at random from *min_port*‒*max_port*.

    This function repeatedly samples a port number from the given range and
    performs a bind-test. It stops at the first port that can be bound
    successfully, guaranteeing exclusivity at the moment of return.

    Args:
        min_port: Inclusive lower bound of the search range.
        max_port: Inclusive upper bound of the search range.
        max_attempts: Maximum number of random probes before aborting.

    Returns:
        A currently unused TCP port.

    Raises:
        ValueError: If the specified range is invalid.
        RuntimeError: If no free port is found after *max_attempts* probes.
    """
    if not (0 < min_port <= max_port < 65_536):
        raise ValueError("Port range must be within 1-65535 and min <= max.")

    for _ in range(max_attempts):
        port = random.randint(min_port, max_port)
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("", port))
                return port  # Port is free at this moment
            except OSError:
                continue  # Port is in use, try another

    raise RuntimeError("Unable to find a free port after multiple attempts.")


def find_free_port_os() -> int:
    """Ask the OS for an unused TCP port and return it.

    The OS selects an ephemeral port by binding to port 0. This is simpler than
    *get_random_free_port* but the chosen value is not truly random.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def find_free_port_in_range(start_port: int, max_range: int = 1000) -> int:
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
