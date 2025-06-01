import logging

import carla

logger = logging.getLogger(__name__)


class ZombieVehicle:
    def __init__(self, actor_id, world):
        self._vehicle = world.get_actor(actor_id)

    def teleport_to(self, transform):
        self._vehicle.set_transform(transform)
        self._vehicle.set_velocity(carla.Vector3D())

    def clean(self):
        if not self._vehicle.is_alive:
            logger.warning(f"Zombie vehicle {self._vehicle.id} is not alive. Trying to destroy it anyway.")
        self._vehicle.destroy()
