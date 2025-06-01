"""Base classes for CARLA actors in the engine system."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import carla
import numpy as np

logger = logging.getLogger(__name__)


class BaseActor(ABC):
    """Abstract base class for all CARLA actors in the engine system."""

    def __init__(self, actor: carla.Actor, world: carla.World):
        """Initialize the base actor.

        Args:
            actor: The CARLA actor instance
            world: The CARLA world instance
        """
        self.actor = actor
        self.world = world
        self._is_alive = True
        self._tick_count = 0

    @property
    def id(self) -> int:
        """Get the actor's unique ID."""
        return self.actor.id

    @property
    def location(self) -> carla.Location:
        """Get the actor's current location."""
        return self.actor.get_location()

    @property
    def transform(self) -> carla.Transform:
        """Get the actor's current transform."""
        return self.actor.get_transform()

    @property
    def velocity(self) -> carla.Vector3D:
        """Get the actor's current velocity."""
        return self.actor.get_velocity()

    @property
    def is_alive(self) -> bool:
        """Check if the actor is still alive."""
        return self._is_alive and self.actor.is_alive

    @abstractmethod
    def tick(self, timestamp: dict[str, Any]) -> dict[str, Any]:
        """Update the actor state for the current simulation step.

        Args:
            timestamp: Simulation timestamp information

        Returns:
            Dictionary containing actor status and metrics
        """
        pass

    def destroy(self) -> None:
        """Clean up and destroy the actor."""
        if self.is_alive:
            try:
                self.actor.destroy()
            except Exception as e:
                logger.warning(f"Failed to destroy actor {self.id}: {e}")
            finally:
                self._is_alive = False

    def __del__(self):
        """Destructor to ensure cleanup."""
        if hasattr(self, "_is_alive") and self._is_alive:
            self.destroy()


class BaseVehicle(BaseActor):
    """Base class for vehicle actors with common vehicle functionality."""

    def __init__(self, vehicle: carla.Vehicle, world: carla.World):
        """Initialize the base vehicle.

        Args:
            vehicle: The CARLA vehicle instance
            world: The CARLA world instance
        """
        if not isinstance(vehicle, carla.Vehicle):
            raise TypeError("Actor must be a carla.Vehicle instance")

        super().__init__(vehicle, world)
        self.vehicle = vehicle  # Type hint for better IDE support
        self._map = world.get_map()

    @property
    def speed_kmh(self) -> float:
        """Get vehicle speed in km/h."""
        velocity = self.velocity
        speed_ms = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        return speed_ms * 3.6

    @property
    def current_waypoint(self) -> carla.Waypoint:
        """Get the waypoint closest to the vehicle's current location."""
        return self._map.get_waypoint(self.location)

    def set_light_state(self, light_state: carla.VehicleLightState) -> None:
        """Set the vehicle's light state.

        Args:
            light_state: The light state to set
        """
        try:
            self.vehicle.set_light_state(light_state)
        except Exception as e:
            logger.warning(f"Failed to set light state for vehicle {self.id}: {e}")

    def get_control(self) -> carla.VehicleControl:
        """Get the vehicle's current control state."""
        return self.vehicle.get_control()

    def apply_control(self, control: carla.VehicleControl) -> None:
        """Apply control commands to the vehicle.

        Args:
            control: The control commands to apply
        """
        try:
            self.vehicle.apply_control(control)
        except Exception as e:
            logger.warning(f"Failed to apply control to vehicle {self.id}: {e}")

    def update_lights_for_weather(self) -> None:
        """Update vehicle lights based on current weather conditions."""
        try:
            weather = self.world.get_weather()
            if weather.sun_altitude_angle < 0.0:
                # Night time - turn on position and low beam lights
                light_state = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
            else:
                # Day time - turn off lights
                light_state = carla.VehicleLightState.NONE

            self.set_light_state(light_state)
        except Exception as e:
            logger.warning(f"Failed to update lights for vehicle {self.id}: {e}")


class BaseNavigationVehicle(BaseVehicle):
    """Base class for vehicles with navigation capabilities."""

    def __init__(
        self,
        vehicle: carla.Vehicle,
        world: carla.World,
        target_locations: list[carla.Location] | None = None,
    ):
        """Initialize the navigation vehicle.

        Args:
            vehicle: The CARLA vehicle instance
            world: The CARLA world instance
            target_locations: List of target locations for navigation
        """
        super().__init__(vehicle, world)

        self._target_locations = target_locations or []
        self._current_target_index = 0
        self._route_completed = 0.0
        self._route_length = 0.0

    @property
    def current_target(self) -> carla.Location | None:
        """Get the current navigation target."""
        if self._current_target_index < len(self._target_locations):
            return self._target_locations[self._current_target_index]
        return None

    @property
    def route_completion_ratio(self) -> float:
        """Get the route completion ratio (0.0 to 1.0)."""
        if self._route_length <= 0:
            return 0.0
        return min(self._route_completed / self._route_length, 1.0)

    def add_target_location(self, location: carla.Location) -> None:
        """Add a new target location to the navigation plan.

        Args:
            location: The target location to add
        """
        self._target_locations.append(location)

    def clear_targets(self) -> None:
        """Clear all target locations."""
        self._target_locations.clear()
        self._current_target_index = 0

    def advance_to_next_target(self) -> bool:
        """Advance to the next target in the navigation plan.

        Returns:
            True if there was a next target to advance to, False otherwise
        """
        if self._current_target_index < len(self._target_locations) - 1:
            self._current_target_index += 1
            return True
        return False

    def is_near_target(self, target: carla.Location, threshold: float = 10.0) -> bool:
        """Check if the vehicle is near the specified target location.

        Args:
            target: The target location to check
            threshold: Distance threshold in meters

        Returns:
            True if the vehicle is within the threshold distance
        """
        return self.location.distance(target) < threshold

    def is_route_completed(self, distance_threshold: float = 10.0) -> bool:
        """Check if the entire route has been completed.

        Args:
            distance_threshold: Distance threshold for considering arrival

        Returns:
            True if the route is completed
        """
        if not self._target_locations:
            return True

        final_target = self._target_locations[-1]
        return self.is_near_target(final_target, distance_threshold)


class ActorManager:
    """Manager for tracking and controlling multiple actors."""

    def __init__(self):
        """Initialize the actor manager."""
        self._actors: dict[int, BaseActor] = {}
        self._actor_groups: dict[str, list[int]] = {}

    def register_actor(self, actor: BaseActor, group: str = "default") -> None:
        """Register an actor with the manager.

        Args:
            actor: The actor to register
            group: The group to assign the actor to
        """
        if actor.id in self._actors:
            logger.warning(f"Actor {actor.id} is already registered")
            return

        self._actors[actor.id] = actor

        if group not in self._actor_groups:
            self._actor_groups[group] = []
        self._actor_groups[group].append(actor.id)

        logger.debug(f"Registered actor {actor.id} in group '{group}'")

    def unregister_actor(self, actor_id: int) -> bool:
        """Unregister an actor from the manager.

        Args:
            actor_id: The ID of the actor to unregister

        Returns:
            True if the actor was found and removed
        """
        if actor_id not in self._actors:
            return False

        # Remove from groups
        for group_actors in self._actor_groups.values():
            if actor_id in group_actors:
                group_actors.remove(actor_id)

        # Remove from main registry
        del self._actors[actor_id]
        logger.debug(f"Unregistered actor {actor_id}")
        return True

    def get_actor(self, actor_id: int) -> BaseActor | None:
        """Get an actor by ID.

        Args:
            actor_id: The actor ID to look up

        Returns:
            The actor instance or None if not found
        """
        return self._actors.get(actor_id)

    def get_actors_in_group(self, group: str) -> list[BaseActor]:
        """Get all actors in a specific group.

        Args:
            group: The group name

        Returns:
            List of actors in the group
        """
        actor_ids = self._actor_groups.get(group, [])
        return [self._actors[actor_id] for actor_id in actor_ids if actor_id in self._actors]

    def get_all_actors(self) -> list[BaseActor]:
        """Get all registered actors.

        Returns:
            List of all actors
        """
        return list(self._actors.values())

    def tick_all(self, timestamp: dict[str, Any]) -> dict[int, dict[str, Any]]:
        """Tick all registered actors.

        Args:
            timestamp: Simulation timestamp information

        Returns:
            Dictionary mapping actor IDs to their tick results
        """
        results: dict[int, dict[str, Any]] = {}
        dead_actors: list[int] = []

        for actor_id, actor in self._actors.items():
            if not actor.is_alive:
                dead_actors.append(actor_id)
                continue

            try:
                result = actor.tick(timestamp)
                results[actor_id] = result
            except Exception as e:
                logger.error(f"Error ticking actor {actor_id}: {e}")
                dead_actors.append(actor_id)

        # Clean up dead actors
        for actor_id in dead_actors:
            self.unregister_actor(actor_id)

        return results

    def destroy_all(self) -> None:
        """Destroy all registered actors."""
        for actor in self._actors.values():
            try:
                actor.destroy()
            except Exception as e:
                logger.error(f"Error destroying actor {actor.id}: {e}")

        self._actors.clear()
        self._actor_groups.clear()
        logger.info("Destroyed all actors")

    def get_stats(self) -> dict[str, Any]:
        """Get manager statistics.

        Returns:
            Dictionary containing manager statistics
        """
        alive_count = sum(1 for actor in self._actors.values() if actor.is_alive)

        group_stats: dict[str, int] = {}
        for group, actor_ids in self._actor_groups.items():
            group_stats[group] = len(actor_ids)

        return {
            "total_actors": len(self._actors),
            "alive_actors": alive_count,
            "dead_actors": len(self._actors) - alive_count,
            "groups": group_stats,
        }
