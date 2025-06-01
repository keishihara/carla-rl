from __future__ import annotations

import logging

import carla
import numpy as np

from carla_env.v2.engine.actors.npc.base import NpcActor

logger = logging.getLogger(__name__)


class NpcWalker(NpcActor):
    """NPC walker actor."""

    def __init__(
        self,
        actor_id: int,
        controller_id: int,
        world: carla.World,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(actor_id=actor_id, world=world, rng=rng)

        self._controller = world.get_actor(controller_id)
        self._controller.start()
        self._controller.go_to_location(world.get_random_location_from_navigation())
        self._controller.set_max_speed(1 + self._rng.random())

    def clean(self) -> None:
        """Clean up the actor."""
        self._controller.stop()
        self._controller.destroy()

        super().clean()


class NpcWalkerHandler:
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
        self._npc_walkers = {}

    def reset(
        self,
        num_npc_walkers: int | list[int],
        ev_spawn_locations: list[carla.Transform],
        rng: np.random.Generator | None = None,
    ) -> None:
        self._rng = rng or self._rng

        if isinstance(num_npc_walkers, list):
            n_spawn = self._rng.integers(num_npc_walkers[0], num_npc_walkers[1] + 1)
        else:
            n_spawn = num_npc_walkers

        self._spawn(n_spawn, ev_spawn_locations)
        logger.debug(f"Spawned {len(self._npc_walkers)} NPC walkers. Should Spawn {num_npc_walkers}")

    def _spawn(
        self,
        num_npc_walkers: int,
        ev_spawn_locations: list[carla.Transform],
        max_trial: int = 10,
        tick: bool = True,
    ) -> None:
        SpawnActor = carla.command.SpawnActor
        walker_bp_library = self._world.get_blueprint_library().filter("walker.pedestrian.*")
        walker_controller_bp = self._world.get_blueprint_library().find("controller.ai.walker")

        def proximity_to_ev(location):
            return any([ev_loc.distance(location) < self._spawn_distance_to_ev for ev_loc in ev_spawn_locations])

        controller_ids = []
        walker_ids = []
        num_spawned = 0
        n_trial = 0
        while num_spawned < num_npc_walkers:
            spawn_points = []
            _walkers = []
            _controllers = []

            for i in range(num_npc_walkers - num_spawned):
                is_proximity_to_ev = True
                spawn_loc = None
                while is_proximity_to_ev:
                    spawn_loc = self._world.get_random_location_from_navigation()
                    if spawn_loc is not None:
                        is_proximity_to_ev = proximity_to_ev(spawn_loc)
                spawn_points.append(carla.Transform(location=spawn_loc))

            batch = []
            for spawn_point in spawn_points:
                walker_bp = np.random.choice(walker_bp_library)
                if walker_bp.has_attribute("is_invincible"):
                    walker_bp.set_attribute("is_invincible", "false")
                batch.append(SpawnActor(walker_bp, spawn_point))

            for result in self._client.apply_batch_sync(batch, tick):
                if not result.error:
                    num_spawned += 1
                    _walkers.append(result.actor_id)

            batch = [SpawnActor(walker_controller_bp, carla.Transform(), walker) for walker in _walkers]
            for result in self._client.apply_batch_sync(batch, tick):
                if result.error:
                    logger.error(result.error)
                else:
                    _controllers.append(result.actor_id)

            controller_ids.extend(_controllers)
            walker_ids.extend(_walkers)

            n_trial += 1
            if n_trial == max_trial and (num_npc_walkers - num_spawned) > 0:
                logger.warning(
                    f"{self._world.get_map().name}: "
                    f"Spawning NPC walkers max trial {n_trial} reached! "
                    f"spawned/to_spawn: {num_spawned}/{num_npc_walkers}"
                )
                break

        for w_id, c_id in zip(walker_ids, controller_ids):
            self._npc_walkers[w_id] = NpcWalker(
                actor_id=w_id,
                controller_id=c_id,
                world=self._world,
                rng=self._rng,
            )

        return self._npc_walkers

    def tick(self):
        pass

    def clean(self):
        alive_walker_ids = [walker.id for walker in self._world.get_actors().filter("*walker.pedestrian*")]

        for npc_walker_id, npc_walker in self._npc_walkers.items():
            if npc_walker_id in alive_walker_ids:
                npc_walker.clean()

        self._npc_walkers = {}
