import math

import carla
import numpy as np


def draw_waypoints(world: carla.World, waypoints: list[carla.Waypoint], z: float = 0.5) -> None:
    """Draw a list of waypoints at a certain height given in z.

    Args:
        world: carla.world object.
        waypoints: List or iterable container with the waypoints to draw.
        z: Height in meters.

    Returns:
        None
    """
    for w in waypoints:
        t = w.transform
        begin = t.location + carla.Location(z=z)
        angle = math.radians(t.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=1.0)


def get_speed(vehicle: carla.Vehicle) -> float:
    """Compute speed of a vehicle in Kmh.

    Args:
        vehicle: The vehicle for which speed is calculated.

    Returns:
        float: Speed in Kmh.
    """
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)


def compute_yaw_difference(yaw1: float, yaw2: float) -> float:
    """Compute the difference in yaw between two angles.

    Args:
        yaw1: First yaw angle in degrees.
        yaw2: Second yaw angle in degrees.

    Returns:
        float: The difference in yaw in degrees.
    """
    u = np.array(
        [
            math.cos(math.radians(yaw1)),
            math.sin(math.radians(yaw1)),
        ]
    )

    v = np.array(
        [
            math.cos(math.radians(yaw2)),
            math.sin(math.radians(yaw2)),
        ]
    )

    angle = math.degrees(math.acos(np.clip(np.dot(u, v), -1, 1)))

    return angle


def is_within_distance_ahead(
    target_location: carla.Location,
    current_location: carla.Location,
    orientation: float,
    max_distance: float,
    degree: float = 60,
) -> bool:
    """Check if a target object is within a certain distance in front of a reference object.

    Args:
        target_location: Location of the target object.
        current_location: Location of the reference object.
        orientation: Orientation of the reference object.
        max_distance: Maximum allowed distance.
        degree: Maximum angle in degrees to consider the target ahead.

    Returns:
        bool: True if the target object is within max_distance ahead of the reference object.
    """
    u = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    distance = np.linalg.norm(u)

    if distance > max_distance:
        return False

    v = np.array([math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])

    angle = math.degrees(math.acos(np.dot(u, v) / distance))

    return angle < degree


def compute_magnitude_angle(
    target_location: carla.Location,
    current_location: carla.Location,
    orientation: float,
) -> tuple[float, float]:
    """Compute relative angle and distance between a target_location and a current_location.

    Args:
        target_location: Location of the target object.
        current_location: Location of the reference object.
        orientation: Orientation of the reference object.

    Returns:
        tuple: A tuple composed of the distance to the object and the angle between both objects.
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    forward_vector = np.array([math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))

    return (norm_target, d_angle)


def distance_vehicle(waypoint: carla.Waypoint, vehicle_transform: carla.Transform) -> float:
    """Calculate the distance between a waypoint and a vehicle.

    Args:
        waypoint: The waypoint to measure from.
        vehicle_transform: The transform of the vehicle.

    Returns:
        float: The distance between the waypoint and the vehicle.
    """
    loc = vehicle_transform.location
    dx = waypoint.transform.location.x - loc.x
    dy = waypoint.transform.location.y - loc.y

    return math.sqrt(dx * dx + dy * dy)


def vector(location_1: carla.Location, location_2: carla.Location) -> list[float]:
    """Return the unit vector from location_1 to location_2.

    Args:
        location_1: The starting carla.Location object.
        location_2: The ending carla.Location object.

    Returns:
        list: The unit vector from location_1 to location_2.
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z])

    return [x / norm, y / norm, z / norm]
