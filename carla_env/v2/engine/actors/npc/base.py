from __future__ import annotations

import logging

import carla
import numpy as np

logger = logging.getLogger(__name__)


class NpcActor:
    """Base class for all NPC actors (vehicles and walkers)."""

    def __init__(self, actor_id: int, world: carla.World, rng: np.random.Generator | None = None):
        self._actor_id = actor_id
        self._world = world
        self._actor = world.get_actor(actor_id)
        self._rng = rng if rng is not None else np.random.default_rng()

    def teleport_to(self, transform: carla.Transform) -> None:
        """Teleport actor to specified transform."""
        raise NotImplementedError

    def clean(self) -> None:
        """Clean up the actor."""
        if not self._actor.is_alive:
            logger.warning(f"NPC actor {self._actor.id} is not alive. Trying to destroy it anyway.")
        self._actor.destroy()
