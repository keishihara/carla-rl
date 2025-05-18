import random
import time
from typing import List

import carla


def spawn_random_vehicles(host: str = "localhost", port: int = 2000, num_vehicles: int = 10) -> List[carla.Vehicle]:
    """Spawn a given number of vehicles at random locations in the Carla world.

    Args:
        host: The hostname or IP of the Carla server.
        port: The port of the Carla server.
        num_vehicles: Number of vehicles to spawn.

    Returns:
        List of spawned Vehicle actors.
    """
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()

    blueprint_library = world.get_blueprint_library().filter("vehicle.*")
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    spawned: List[carla.Vehicle] = []
    for transform in spawn_points[:num_vehicles]:
        bp = random.choice(blueprint_library)
        vehicle = world.try_spawn_actor(bp, transform)
        if vehicle:
            spawned.append(vehicle)

    print(f"Spawned {len(spawned)}/{num_vehicles} vehicles")
    return spawned


def measure_carla_fps(host: str = "localhost", port: int = 2000, frame_count: int = 1000, n_vehicles: int = 10) -> None:
    """Measure and print the simulation FPS after spawning vehicles.

    Args:
        host: Carla server hostname or IP.
        port: Carla server port.
        frame_count: Number of ticks to simulate.
        n_vehicles: Number of vehicles to spawn.
    """
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()

    vehicles = spawn_random_vehicles(host, port, num_vehicles=n_vehicles)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / 20.0
    world.apply_settings(settings)

    start = time.time()
    for _ in range(frame_count):
        world.tick()
    end = time.time()

    elapsed = end - start
    fps = frame_count / elapsed
    print(f"Measured FPS over {frame_count} ticks with {len(vehicles)} vehicles: {fps:.2f}")

    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)


if __name__ == "__main__":
    measure_carla_fps(n_vehicles=20)
