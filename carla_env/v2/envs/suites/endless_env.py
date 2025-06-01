from __future__ import annotations

from carla_env.v2.envs.carla_env import CarlaEnv


class EndlessEnv(CarlaEnv):
    def __init__(
        self,
        map_name: str = "Town01",
        host: str = "localhost",
        seed: int = 2025,
        gpu_id: int = 0,
        no_rendering: bool = False,
        nullrhi: bool = False,
        obs_configs: dict | None = None,
        reward_configs: dict | None = None,
        terminal_configs: dict | None = None,
        weather_group: str = "dynamic_1.0",
        render_mode: str = "rgb_array",
        num_npc_vehicles: int | list[int] | None = None,
        num_npc_walkers: int | list[int] | None = None,
    ):
        # Set default obs_configs if not provided
        if obs_configs is None:
            obs_configs = self._get_default_obs_configs()

        # Build all tasks using the static method
        all_tasks = self.build_all_tasks(num_npc_vehicles, num_npc_walkers, weather_group)

        super().__init__(
            map_name=map_name,
            host=host,
            seed=seed,
            gpu_id=gpu_id,
            no_rendering=no_rendering,
            nullrhi=nullrhi,
            obs_configs=obs_configs,
            reward_configs=reward_configs or self._get_default_reward_configs(),
            terminal_configs=terminal_configs or self._get_default_terminal_configs(),
            all_tasks=all_tasks,
            render_mode=render_mode,
        )

        self.weather_group = weather_group
        self.num_npc_vehicles = num_npc_vehicles
        self.num_npc_walkers = num_npc_walkers

    @staticmethod
    def build_all_tasks(
        num_npc_vehicles: int | list[int] | None, num_npc_walkers: int | list[int] | None, weather_group: str
    ) -> dict:
        """Build task configurations based on weather group.

        Args:
            num_npc_vehicles: Single value or [min, max] range
            num_npc_walkers: Single value or [min, max] range
            weather_group: Weather configuration (e.g., "dynamic_1.0", "train", "all")
        """
        if num_npc_vehicles is None:
            num_npc_vehicles = [0, 50]
        if num_npc_walkers is None:
            num_npc_walkers = [0, 50]

        # Handle dynamic weather or predefined weather groups
        if "dynamic" in weather_group:
            # For dynamic weather, create single task
            weathers = [weather_group]
        elif weather_group == "new":
            weathers = ["SoftRainSunset", "WetSunset"]
        elif weather_group == "train":
            weathers = ["ClearNoon", "WetNoon", "HardRainNoon", "ClearSunset"]
        elif weather_group == "all":
            weathers = [
                "ClearNoon",
                "CloudyNoon",
                "WetNoon",
                "WetCloudyNoon",
                "SoftRainNoon",
                "MidRainyNoon",
                "HardRainNoon",
                "ClearSunset",
                "CloudySunset",
                "WetSunset",
                "WetCloudySunset",
                "SoftRainSunset",
                "MidRainSunset",
                "HardRainSunset",
            ]
        else:
            # Single weather condition
            weathers = [weather_group]

        all_tasks = {}
        for task_idx, weather in enumerate(weathers):
            task = {
                "weather": weather,
                "description_folder": "None",
                "route_id": 0,
                "num_zombie_vehicles": num_npc_vehicles,
                "num_zombie_walkers": num_npc_walkers,
                # EgoVehicleHandler expects these at the top level of ego_vehicles
                "ego_vehicles": {
                    "actors": {
                        "hero": {"model": "vehicle.lincoln.mkz_2017"}
                    },  # EgoVehicleHandler expects task_config["actors"]
                    "routes": {"hero": []},  # EgoVehicleHandler expects task_config["routes"] (empty for endless)
                    "endless": {"hero": True},  # EgoVehicleHandler expects task_config.get("endless")
                },
                "scenario_actors": {},
            }
            all_tasks[task_idx] = task

        return all_tasks

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple:
        """Reset environment with dynamic NPC counts if ranges are specified."""
        # Handle dynamic NPC counts
        self._update_npc_counts()

        # Call parent reset
        return super().reset(seed=seed, options=options)

    def _update_npc_counts(self) -> None:
        """Update NPC counts in current task if ranges are specified."""
        if not hasattr(self, "_task") or self._task is None:
            return

        # Update NPC vehicles count
        if isinstance(self.num_npc_vehicles, list) and len(self.num_npc_vehicles) == 2:
            min_vehicles, max_vehicles = self.num_npc_vehicles
            self._task["num_zombie_vehicles"] = self._rng.integers(min_vehicles, max_vehicles + 1)

        # Update NPC walkers count
        if isinstance(self.num_npc_walkers, list) and len(self.num_npc_walkers) == 2:
            min_walkers, max_walkers = self.num_npc_walkers
            self._task["num_zombie_walkers"] = self._rng.integers(min_walkers, max_walkers + 1)

    def get_available_weather_groups(self) -> list[str]:
        """Get list of available weather groups."""
        return ["dynamic_1.0", "dynamic_2.0", "new", "train", "all"]

    def get_current_weather_group(self) -> str:
        """Get current weather group."""
        return self.weather_group

    def get_weather_count(self) -> int:
        """Get number of weather conditions in current group."""
        return len(self._all_tasks)

    def get_current_npc_counts(self) -> dict:
        """Get current NPC counts for the active task."""
        if hasattr(self, "_task") and self._task is not None:
            return {
                "npc_vehicles": self._task.get("num_zombie_vehicles", 0),
                "npc_walkers": self._task.get("num_zombie_walkers", 0),
            }
        return {"npc_vehicles": 0, "npc_walkers": 0}

    @staticmethod
    def _get_default_obs_configs() -> dict:
        """Get default observation configurations based on birdview_no_scale.yaml."""
        return {
            "hero": {
                "birdview": {
                    "module": "birdview.chauffeurnet",
                    "width_in_pixels": 192,
                    "pixels_ev_to_bottom": 40,
                    "pixels_per_meter": 5.0,
                    "history_idx": [-16, -11, -6, -1],
                    "scale_bbox": False,
                    "scale_mask_col": 1.1,
                },
                "speed": {
                    "module": "actor_state.speed",
                },
                "control": {
                    "module": "actor_state.control",
                },
                "velocity": {
                    "module": "actor_state.velocity",
                },
            }
        }

    @staticmethod
    def _get_default_reward_configs() -> dict:
        return {
            "hero": {
                "entry_point": "reward.valeo_action:ValeoAction",
                "kwargs": {},
            }
        }

    @staticmethod
    def _get_default_terminal_configs() -> dict:
        return {
            "hero": {
                "entry_point": "terminal.valeo_no_det_px:ValeoNoDetPx",
                "kwargs": {},
            }
        }
