from __future__ import annotations

import importlib
from typing import Mapping

from gymnasium import spaces

OBS_PACKAGE_ROOT = "carla_env.engine.sensors"


class Manager:
    """Orchestrates multiple *observation managers* per ego vehicle.

    Each ego vehicle can host several sensor managers (RGB camera, BEV, …).
    This class creates them from config, forwards :py:meth:`get_observation`,
    and aggregates gym spaces.

    Args:
        obs_configs: Nested config dict::

            {
                \"ego_vehicle_id\": {
                    \"camera_front\": {\"module\": \"camera.rgb\", ...},
                    \"bev\": {\"module\": \"birdview.chauffeurnet\", ...},
                },
                ...
            }
    """

    def __init__(self, obs_configs: Mapping[str, Mapping[str, dict]]) -> None:
        self._obs_configs = obs_configs
        self._managers: dict[str, dict[str, object]] = {}
        self._build_managers()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_observation(self) -> dict:
        obs: dict[str, dict[str, object]] = {}
        for ev_id, mgr_dict in self._managers.items():
            obs[ev_id] = {obs_id: mgr.get_observation() for obs_id, mgr in mgr_dict.items()}
        return obs

    @property
    def obs_space(self) -> spaces.Dict:
        """Return a nested gymnasium space describing all observations."""
        ev_spaces: dict[str, spaces.Dict] = {
            ev_id: self._build_space(mgr_dict) for ev_id, mgr_dict in self._managers.items()
        }
        return spaces.Dict(ev_spaces)

    def attach_ego_vehicles(self, ego_actors: Mapping[str, object]) -> None:
        """Attach freshly spawned ego-vehicle actors to each manager."""
        for ev_id, actor in ego_actors.items():
            for mgr in self._managers[ev_id].values():
                mgr.attach_ego_vehicle(actor)

    def reset(self) -> None:
        """Re-instantiate managers from the original config."""
        self.clean()
        self._build_managers()

    def clean(self) -> None:
        for mgr_dict in self._managers.values():
            for mgr in mgr_dict.values():
                mgr.clean()
        self._managers.clear()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _build_managers(self) -> None:
        """Instantiate managers as specified in ``self._obs_configs``."""
        for ev_id, cfgs in self._obs_configs.items():
            self._managers[ev_id] = {}
            for obs_id, cfg in cfgs.items():
                self._managers[ev_id][obs_id] = self._create_manager(cfg)

    def _create_manager(self, cfg: dict):
        module_path = f"{OBS_PACKAGE_ROOT}.{cfg['module']}"
        try:
            mod = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"Failed to import '{module_path}'. Is the module path correct?"  # noqa: TRY003
            ) from exc
        return mod.ObsManager(cfg)

    @staticmethod
    def _build_space(mgr_dict: Mapping[str, object]) -> spaces.Dict:
        return spaces.Dict({obs_id: mgr.obs_space for obs_id, mgr in mgr_dict.items()})
