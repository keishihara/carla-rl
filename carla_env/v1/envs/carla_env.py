from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
import warnings
from typing import Literal, Sequence

import carla
import gymnasium as gym
import numpy as np
from PIL import Image

from carla_env.v1.core.obs_manager.obs_manager_handler import ObsManagerHandler
from carla_env.v1.core.task_actor.ego_vehicle.ego_vehicle_handler import EgoVehicleHandler
from carla_env.v1.core.task_actor.scenario_actor.scenario_actor_handler import ScenarioActorHandler
from carla_env.v1.core.zombie_vehicle.zombie_vehicle_handler import ZombieVehicleHandler
from carla_env.v1.core.zombie_walker.zombie_walker_handler import ZombieWalkerHandler
from carla_env.v1.utils.dynamic_weather import WeatherHandler
from carla_env.v1.utils.traffic_light import TrafficLightHandler
from carla_env.v2.runtime.carla_runtime import CarlaRuntime

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class CarlaEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

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
        all_tasks: dict | None = None,
        render_mode: Literal["human", "rgb_array"] = "rgb_array",
    ):
        self._map_name = map_name
        self._host = host
        self._seed = seed
        self._gpu_id = gpu_id
        self._no_rendering = no_rendering
        self._nullrhi = nullrhi

        # Set configs and tasks. If not provided, use empty dict.
        self._obs_configs = obs_configs or {}
        self._reward_configs = reward_configs or {}
        self._terminal_configs = terminal_configs or {}
        self._all_tasks = all_tasks or {}

        self._observation_space = None
        self._action_space = None

        # FPS tracking
        self._step_times = []
        self._last_fps_log = 0

        # Public attributes
        self.name = self.__class__.__name__
        self.render_mode = render_mode

        self._setup()

    @property
    def observation_space(self) -> gym.spaces.Dict:
        if self._observation_space is None:
            raise ValueError("observation_space is not initialized yet.")
        return self._observation_space

    @observation_space.setter
    def observation_space(self, observation_space: gym.spaces.Dict) -> None:
        self._observation_space = observation_space

    @property
    def action_space(self) -> gym.spaces.Dict:
        if self._action_space is None:
            raise ValueError("action_space is not initialized yet.")
        return self._action_space

    @action_space.setter
    def action_space(self, action_space: gym.spaces.Dict) -> None:
        self._action_space = action_space

    @property
    def num_tasks(self) -> int:
        return len(self._all_tasks)

    @property
    def task(self) -> dict:
        return self._task

    @property
    def obs_configs(self) -> dict:
        return self._obs_configs

    @property
    def port(self) -> int:
        return self._runtime.ports.rpc_port

    @property
    def tm_port(self) -> int:
        return self._runtime.ports.tm_port

    @property
    def timestamp(self):
        if not hasattr(self, "_timestamp"):
            raise ValueError("timestamp is not set. Please call reset() first.")
        return self._timestamp.copy()

    @timestamp.setter
    def timestamp(self, value: dict) -> None:
        self._timestamp = value

    def set_task_idx(self, task_idx):
        self._task_idx = task_idx
        self._shuffle_task = False
        self._task = self._all_tasks[self._task_idx].copy()

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple:
        """Reset the environment and return the first observation."""
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(self._seed)

        options = options or {}

        if "gpu_id" in options:
            self._gpu_id = options["gpu_id"]
            self._runtime.set_gpu_id(options["gpu_id"])

        if "no_rendering_mode" in options:
            self._runtime.set_no_rendering_mode(options["no_rendering_mode"], immediate=True)

        if "nullrhi" in options:
            self._runtime.set_nullrhi(options["nullrhi"])

        # Maintain server / client
        server_restarted = self._runtime.ensure_healthy()
        if server_restarted:
            # タイマーをリセット
            for attr in ("_timestamp", "_world_time"):
                if hasattr(self, attr):
                    delattr(self, attr)

        # Task shuffle
        if self._shuffle_task:
            self._task_idx = self._rng.choice(self.num_tasks)
            self._task = self._all_tasks[self._task_idx].copy()

        # Handlers
        self._reset_handlers(server_restarted=server_restarted)

        # First tick of the new episode
        self._runtime.world.tick()
        self._update_timestamp(reset_called=True)

        # Observation
        reward_d, terminated_d, truncated_d, info_d = self._ev_handler.tick(self.timestamp)
        obs = self._get_observation()
        info = {
            "timestamp": self._timestamp,
            "task": self._task,
            "reward": reward_d,
            "terminated": terminated_d,
            "truncated": truncated_d,
            "info": info_d,
        }
        return obs, info

    def step(self, control_dict: dict) -> tuple:
        """Run one control step in the simulation."""
        step_start = time.time()

        # Apply control
        control_dict = self._process_action(control_dict)
        self._ev_handler.apply_control(control_dict)
        self._sa_handler.tick()

        # Simulator tick
        self._runtime.world.tick()
        self._update_timestamp()

        # Compute reward / terminated / truncated / info
        reward_d, terminated_d, truncated_d, info_d = self._ev_handler.tick(self.timestamp)

        # Observation
        obs_d = self._get_observation()

        # Weather update
        self._wt_handler.tick(self._timestamp["delta_seconds"])

        # FPS tracking
        step_time = time.time() - step_start
        self._step_times.append(step_time)

        # Log FPS every 100 steps
        if len(self._step_times) >= 100:
            avg_step_time = sum(self._step_times) / len(self._step_times)
            fps = 1.0 / avg_step_time if avg_step_time > 0 else 0
            logger.info(f"Average FPS over last {len(self._step_times)} steps: {fps:.2f}")
            self._step_times = []  # Reset for next measurement

        # Logging
        self._log(obs_d, reward_d, terminated_d, truncated_d, info_d)

        return obs_d, reward_d, terminated_d, truncated_d, info_d

    def close(self) -> None:
        """Release resources; safe to call multiple times."""
        with contextlib.suppress(Exception):
            self._clean()
        with contextlib.suppress(Exception):
            self._runtime.close()

    def render(self):
        """Return or display the latest RGB frame.

        Returns:
            numpy.ndarray | None
                - If ``self.render_mode == "rgb_array"``, returns an ``(H, W, 3)``
                    ``uint8`` array representing the most recent frame.
                - If ``self.render_mode == "human"``, the frame is shown in a
                    blocking window (PIL) and the method returns ``None``.
        """
        if not hasattr(self, "_latest_frame"):
            raise RuntimeError("No frame available yet. Call reset() or step() first.")

        if self.render_mode == "rgb_array":
            return self._latest_frame

        if self.render_mode == "human":
            # For human mode, display the birdview image if available
            if isinstance(self._latest_frame, dict) and "hero" in self._latest_frame:
                if "birdview" in self._latest_frame["hero"] and "rendered" in self._latest_frame["hero"]["birdview"]:
                    Image.fromarray(self._latest_frame["hero"]["birdview"]["rendered"]).show()
                else:
                    logger.warning("No birdview image available for display")
            return None

        raise NotImplementedError(f"Unsupported render_mode: {self.render_mode}")

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        self.close()

    def _setup(self) -> None:
        """Start the CARLA runtime (server+client) and bootstrap handlers."""
        self._runtime = CarlaRuntime(
            map_name=self._map_name,
            gpu_id=0,  # 必要なら引数で受け取る
            no_rendering_mode=self._no_rendering,
            server_cfg=dict(
                rpc_port="auto",
                streaming_port="auto",
                tm_port="auto",
                server_life_time=2400,  # 40 min
                nullrhi=self._nullrhi,
            ),
            client_cfg=dict(
                host=self._host,
                seed=self._seed,
                timeout=120.0,
            ),
        )

        logger.info(f"Starting CARLA runtime with map: {self._map_name}")
        self._runtime.start()

        # RNG
        self._rng = np.random.default_rng(self._seed)

        # First task
        self._task_idx = 0
        self._shuffle_task = True
        self._task = self._all_tasks[self._task_idx]

        # Handlers
        self._init_handlers()

        # Space definitions
        self._observation_space = self._om_handler.observation_space
        if not self._obs_configs:  # single-vehicle fallback
            warnings.warn("obs_configs is empty; using a single Box action space.", stacklevel=2)
            self._action_space = gym.spaces.Box(
                low=np.array([0.0, -1.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self._action_space = gym.spaces.Dict(
                {
                    ego_id: gym.spaces.Box(
                        low=np.array([0.0, -1.0, 0.0], dtype=np.float32),
                        high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                        dtype=np.float32,
                    )
                    for ego_id in self._obs_configs.keys()
                }
            )

        # Async PNG logger (non-blocking I/O)
        self._log_q: queue.Queue[tuple[np.ndarray, str]] = queue.Queue()
        self._logger_thread = threading.Thread(target=self._async_logger_loop, daemon=True)
        self._logger_thread.start()

    def _init_handlers(self) -> None:
        self._om_handler = ObsManagerHandler(self._obs_configs)
        self._ev_handler = EgoVehicleHandler(self._runtime.client, self._reward_configs, self._terminal_configs)
        self._zw_handler = ZombieWalkerHandler(self._runtime.client)
        self._zv_handler = ZombieVehicleHandler(self._runtime.client, tm_port=self._runtime.tm.get_port())
        self._sa_handler = ScenarioActorHandler(self._runtime.client)
        self._wt_handler = WeatherHandler(self._runtime.world)

    def _truncate_num_actors(self) -> None:
        """Adjust NPC counts so they never exceed available spawn slots."""

        def _requested(num_actors: int | Sequence) -> int:
            return num_actors[1] if isinstance(num_actors, list) else num_actors

        total_requested = _requested(self._task["num_zombie_vehicles"]) + _requested(self._task["num_zombie_walkers"])

        world = self._runtime.world
        spawn_points = world.get_map().get_spawn_points()
        total_spawn = len(spawn_points)

        veh_cnt = len(world.get_actors().filter("*vehicle*"))
        wlk_cnt = len(world.get_actors().filter("*walker.pedestrian*"))
        ego_cnt = len(getattr(self._ev_handler, "ego_vehicles", {}))

        occupied = veh_cnt + wlk_cnt + ego_cnt
        free_slots = max(total_spawn - occupied, 0)

        if free_slots == 0:
            warnings.warn("No free spawn points left; forcing NPC counts to 0", stacklevel=2)
            self._task["num_zombie_vehicles"] = 0
            self._task["num_zombie_walkers"] = 0
            return

        if total_requested <= free_slots:
            return

        # Assign the remaining slots to Vehicle / Walker in 60% : 40% ratio
        veh_budget = int(free_slots * 0.6)
        wlk_budget = free_slots - veh_budget  # Remaining slots are assigned to Walker

        def _clamp(val: int | list, budget: int) -> int | list:
            """Clamp int or [min,max] list to ≤ budget."""
            if isinstance(val, Sequence):
                val[0] = max(0, min(val[0], budget))
                val[1] = max(val[0], min(val[1], budget))
                return val
            return max(0, min(val, budget))

        self._task["num_zombie_vehicles"] = _clamp(self._task["num_zombie_vehicles"], veh_budget)
        self._task["num_zombie_walkers"] = _clamp(self._task["num_zombie_walkers"], wlk_budget)

    def _reset_handlers(self, server_restarted: bool = False) -> None:
        if server_restarted:
            self._clean_handlers()
            self._init_handlers()

        self._truncate_num_actors()
        logger.debug(f"num zombie vehicles: {self._task['num_zombie_vehicles']}")
        logger.debug(f"num zombie walkers: {self._task['num_zombie_walkers']}")

        TrafficLightHandler.reset(self._runtime.world)

        self._wt_handler.reset(self._task["weather"])
        ev_spawn_locations = self._ev_handler.reset(self._task["ego_vehicles"])
        self._sa_handler.reset(self._task["scenario_actors"], self._ev_handler.ego_vehicles)
        self._zw_handler.reset(self._task["num_zombie_walkers"], ev_spawn_locations)
        self._zv_handler.reset(self._task["num_zombie_vehicles"], ev_spawn_locations)
        self._om_handler.reset(self._ev_handler.ego_vehicles)

    def _update_timestamp(self, *, reset_called: bool = False) -> None:
        # FIXME: このメソッドはTimerManagerに移行する
        """Update both world-local and env-global timers.

        This method **must** be invoked exactly once per CARLA tick.
        It fuses two internal timing scopes into a public ``self._timestamp`` dict.

        * World-local timer — Resets whenever the CARLA server restarts.
            Keys produced: ``step``, ``frame``, ``wall_time``,
            ``relative_wall_time``, ``simulation_time``,
            ``relative_simulation_time``, ``delta_seconds``.
        * Env-global timer — Lives for the entire lifetime of the :class:`CarlaEnv` instance.
            Keys produced: ``global_step``, ``global_relative_wall_time``.

        Args:
            reset_called (bool, optional):
                Set to ``True`` when the call directly follows
                :py:meth:`reset`. In that case the **per-episode**
                ``step`` counter is re-zeroed, while all global counters
                keep increasing.

        Notes:
            * Internal helpers ``_world_time``, ``_env_time`` and
            ``_prev_sim_time`` are implementation details.
            * The caller is responsible for avoiding concurrent invocations.
        """
        snap = self._runtime.world.get_snapshot()

        # ---------- World-local timer (resets on server restart) ----------
        if not hasattr(self, "_world_time"):
            self._world_time = {
                "start_frame": snap.timestamp.frame,
                "episode_start_frame": snap.timestamp.frame,
                "start_wall_time": snap.timestamp.platform_timestamp,
                "start_sim_time": snap.timestamp.elapsed_seconds,
            }

        if reset_called:
            # Zero the per-episode step counter.
            self._world_time["episode_start_frame"] = snap.timestamp.frame

        world_step = snap.timestamp.frame - self._world_time["episode_start_frame"]
        rel_wall = snap.timestamp.platform_timestamp - self._world_time["start_wall_time"]
        rel_sim = snap.timestamp.elapsed_seconds - self._world_time["start_sim_time"]

        # Simulation seconds since previous tick
        if hasattr(self, "_prev_sim_time"):
            delta_sec = snap.timestamp.elapsed_seconds - self._prev_sim_time
        else:
            delta_sec = 0.0
        self._prev_sim_time = snap.timestamp.elapsed_seconds

        # ---------- Env-global timer (never resets) ----------
        if not hasattr(self, "_env_time"):
            self._env_time = {
                "start_wall_time": snap.timestamp.platform_timestamp,
                "global_step": 0,
            }
        self._env_time["global_step"] += 1
        global_rel_wall = snap.timestamp.platform_timestamp - self._env_time["start_wall_time"]

        # ---------- Consolidated public dict ----------
        self._timestamp = {
            # World-local metrics
            "step": world_step,
            "frame": snap.timestamp.frame,
            "relative_wall_time": rel_wall,
            "wall_time": snap.timestamp.platform_timestamp,
            "relative_simulation_time": rel_sim,
            "simulation_time": snap.timestamp.elapsed_seconds,
            "delta_seconds": delta_sec,
            # Env-global metrics
            "global_step": self._env_time["global_step"],
            "global_relative_wall_time": global_rel_wall,
            # carla_gym compatibility
            "start_frame": self._world_time["start_frame"],
            "start_simulation_time": self._world_time["start_sim_time"],
        }

    def _get_observation(self) -> dict:
        obs = self._om_handler.get_observation(self.timestamp)

        # Set latest frame for rendering
        if "hero" in obs and "birdview" in obs["hero"]:
            self._latest_frame = obs
        else:
            # Fallback: search for the first (H,W,3) image in the dictionary
            warnings.warn(
                "No birdview image found in the observation. Using the first (H,W,3) image in the dictionary.",
                stacklevel=2,
            )
            for agent_obs in obs.values():
                for v in agent_obs.values():
                    if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[2] == 3:
                        self._latest_frame = {"hero": {"birdview": {"rendered": v}}}
                        break
        return obs

    def _process_action(self, control_dict: dict) -> dict:
        """Convert action dict to carla.VehicleControl objects.

        Args:
            control_dict: Dictionary mapping ego_id to action.
                Action can be numpy.ndarray [throttle, steer, brake] or carla.VehicleControl object.

        Returns:
            Dictionary mapping ego_id to carla.VehicleControl objects.
        """
        processed_control_dict = {}

        for ego_id, action in control_dict.items():
            if isinstance(action, np.ndarray):
                # Convert numpy array [throttle, steer, brake] to carla.VehicleControl
                if len(action) != 3:
                    raise ValueError(f"Action array must have length 3, got {len(action)}")

                throttle = float(np.clip(action[0], 0.0, 1.0))
                steer = float(np.clip(action[1], -1.0, 1.0))
                brake = float(np.clip(action[2], 0.0, 1.0))

                processed_control_dict[ego_id] = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
            elif isinstance(action, carla.VehicleControl):
                # Already a VehicleControl object
                processed_control_dict[ego_id] = action
            else:
                raise TypeError(
                    f"Unsupported action type: {type(action)}. Expected numpy.ndarray or carla.VehicleControl"
                )

        return processed_control_dict

    def _log(
        self, obs_dict: dict, reward_dict: dict, terminated_dict: dict, truncated_dict: dict, info_dict: dict
    ) -> None:
        """Queue PNG save and print progress every 1 k global steps."""
        if self._timestamp["global_step"] % 1_000 > 0:
            return

        frame = obs_dict["hero"]["birdview"]["rendered"]
        fname = f"birdview_{self._map_name}_{self._timestamp['global_step']:08d}.png"
        self._log_q.put((frame.copy(), fname))

        completed = info_dict["hero"]["route_completion"]["route_completed_in_m"]
        length = info_dict["hero"]["route_completion"]["route_length_in_m"]
        ratio = completed / max(length, 1e-6) * 100
        logger.info(
            f"[gstep={self._timestamp['global_step']:,}] completed {completed:.2f}/{length:.2f} m ({ratio:.2f} %)"
        )

    def _async_logger_loop(self) -> None:
        """Background worker that writes PNG files without blocking `step()`."""
        while True:
            frame, fname = self._log_q.get()
            try:
                Image.fromarray(frame).save(fname)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"[AsyncLogger] Failed to save {fname}: {exc}")
            self._log_q.task_done()

    def _clean_handlers(self) -> None:
        for _h_name in ("_om_handler", "_ev_handler", "_sa_handler", "_zw_handler", "_zv_handler", "_wt_handler"):
            if not hasattr(self, _h_name):
                continue
            try:
                getattr(self, _h_name).clean()
            except Exception:
                pass

    def _clean(self) -> None:
        self._clean_handlers()
        self._runtime.world.tick()
