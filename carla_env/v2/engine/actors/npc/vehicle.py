from __future__ import annotations

import logging

import carla
import numpy as np

from carla_env.v2.engine.actors.npc.base import NpcActor

logger = logging.getLogger(__name__)


class NpcVehicle(NpcActor):
    """NPC vehicle actor."""


class NpcVehicleHandler:
    def __init__(
        self,
        client: carla.Client,
        tm_port: int,
        spawn_distance_to_ev: float = 10.0,
        rng: np.random.Generator | None = None,
    ):
        self._client = client
        self._world = client.get_world()
        self._spawn_distance_to_ev = spawn_distance_to_ev
        self._tm_port = tm_port

        self._rng = rng if rng is not None else np.random.default_rng()
        self._npc_vehicles = {}

    def reset(
        self,
        num_npc_vehicles: int | list[int],
        ev_spawn_locations: list[carla.Transform],
        rng: np.random.Generator | None = None,
    ) -> None:
        self._rng = rng or self._rng

        if isinstance(num_npc_vehicles, list):
            n_spawn = self._rng.integers(num_npc_vehicles[0], num_npc_vehicles[1] + 1)
        else:
            n_spawn = num_npc_vehicles

        filtered_spawn_points = self._filter_spawn_points(ev_spawn_locations)
        self._rng.shuffle(filtered_spawn_points)

        self._spawn(filtered_spawn_points[:n_spawn])

    def _filter_spawn_points(self, ev_spawn_locations: list[carla.Transform]) -> list[carla.Transform]:
        all_spawn_points = self._world.get_map().get_spawn_points()

        def proximity_to_ev(transform):
            return any(
                [ev_loc.distance(transform.location) < self._spawn_distance_to_ev for ev_loc in ev_spawn_locations]
            )

        filtered_spawn_points = [transform for transform in all_spawn_points if not proximity_to_ev(transform)]

        return filtered_spawn_points

    def _spawn(self, spawn_transforms: list[carla.Transform]) -> None:
        npc_vehicle_ids = []
        blueprints = self._world.get_blueprint_library().filter("vehicle.*")
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        batch = []
        for transform in spawn_transforms:
            blueprint = self._rng.choice(blueprints)

            if blueprint.has_attribute("color"):
                color = self._rng.choice(blueprint.get_attribute("color").recommended_values)
                blueprint.set_attribute("color", color)

            if blueprint.has_attribute("driver_id"):
                driver_id = self._rng.choice(blueprint.get_attribute("driver_id").recommended_values)
                blueprint.set_attribute("driver_id", driver_id)

            blueprint.set_attribute("role_name", "npc_vehicle")

            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True, self._tm_port)))

        for response in self._client.apply_batch_sync(batch, do_tick=True):
            if not response.error:
                npc_vehicle_ids.append(response.actor_id)

        for npc_vehicle_id in npc_vehicle_ids:
            self._npc_vehicles[npc_vehicle_id] = NpcVehicle(
                actor_id=npc_vehicle_id,
                world=self._world,
                rng=self._rng,
            )

        logger.debug(f"Spawned {len(npc_vehicle_ids)} NPC vehicles. Should spawn {len(spawn_transforms)}")

    def tick(self):
        pass

    def clean(self):
        alive_vehicle_ids = [vehicle.id for vehicle in self._world.get_actors().filter("*vehicle*")]
        for npc_vehicle_id, npc_vehicle in self._npc_vehicles.items():
            if npc_vehicle_id in alive_vehicle_ids:
                npc_vehicle.clean()
        self._npc_vehicles = {}
