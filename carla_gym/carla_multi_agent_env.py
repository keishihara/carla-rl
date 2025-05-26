import logging
import time

import carla
import gymnasium as gym
import numpy as np
from PIL import Image
from stable_baselines3.common.utils import set_random_seed

from .core.obs_manager.obs_manager_handler import ObsManagerHandler
from .core.task_actor.ego_vehicle.ego_vehicle_handler import EgoVehicleHandler
from .core.task_actor.scenario_actor.scenario_actor_handler import ScenarioActorHandler
from .core.zombie_vehicle.zombie_vehicle_handler import ZombieVehicleHandler
from .core.zombie_walker.zombie_walker_handler import ZombieWalkerHandler
from .utils.dynamic_weather import WeatherHandler
from .utils.traffic_light import TrafficLightHandler

logger = logging.getLogger(__name__)


class CarlaMultiAgentEnv(gym.Env):
    def __init__(
        self,
        carla_map,
        host,
        port,
        seed,
        no_rendering,
        obs_configs,
        reward_configs,
        terminal_configs,
        all_tasks,
        server_manager,
        render_mode,
    ):
        self._carla_map = carla_map
        self._all_tasks = all_tasks
        self._obs_configs = obs_configs
        self._reward_configs = reward_configs
        self._terminal_configs = terminal_configs
        self._host = host
        self._port = port
        self._seed = seed
        self._no_rendering = no_rendering
        self._server_manager = server_manager
        self.render_mode = render_mode

        self.name = self.__class__.__name__
        self.setup()

    def setup(self):
        self._init_client(
            self._carla_map,
            self._host,
            self._port,
            seed=self._seed,
            no_rendering=self._no_rendering,
        )

        # define observation spaces exposed to agent
        self._om_handler = ObsManagerHandler(self._obs_configs)
        self._ev_handler = EgoVehicleHandler(self._client, self._reward_configs, self._terminal_configs)
        self._zw_handler = ZombieWalkerHandler(self._client)
        self._zv_handler = ZombieVehicleHandler(self._client, tm_port=self._tm.get_port())
        self._sa_handler = ScenarioActorHandler(self._client)
        self._wt_handler = WeatherHandler(self._world)

        TrafficLightHandler.reset(self._world)  # register traffic lights

        # observation spaces
        self.observation_space = self._om_handler.observation_space
        # define action spaces exposed to agent
        # throttle, steer, brake
        self.action_space = gym.spaces.Dict(
            {
                ego_vehicle_id: gym.spaces.Box(
                    low=np.array([0.0, -1.0, 0.0]),
                    high=np.array([1.0, 1.0, 1.0]),
                    dtype=np.float32,
                )
                for ego_vehicle_id in self._obs_configs.keys()
            }
        )

        self._task_idx = 0
        self._shuffle_task = True
        self._task = self._all_tasks[self._task_idx].copy()
        self._start_time = time.time()

    def set_task_idx(self, task_idx):
        self._task_idx = task_idx
        self._shuffle_task = False
        self._task = self._all_tasks[self._task_idx].copy()

    @property
    def num_tasks(self):
        return len(self._all_tasks)

    @property
    def task(self):
        return self._task

    @property
    def obs_configs(self):
        return self._obs_configs

    @property
    def carla_map(self):
        return self._carla_map

    @property
    def port(self):
        return self._port

    @property
    def om_handler(self):
        return self._om_handler

    @property
    def ev_handler(self):
        return self._ev_handler

    @property
    def zw_handler(self):
        return self._zw_handler

    @property
    def zv_handler(self):
        return self._zv_handler

    @property
    def sa_handler(self):
        return self._sa_handler

    @property
    def wt_handler(self):
        return self._wt_handler

    @property
    def client(self):
        return self._client

    @property
    def world(self):
        return self._world

    def reset(self, *, seed=None, options=None):
        if time.time() - self._start_time > self._server_manager.server_life_time_sec:
            logger.info(
                f"Restarting Carla server! elasped {time.time() - self._start_time} seconds > {self._server_manager.server_life_time_sec}"
            )
            self.close()
            self._server_manager.restart()
            self.setup()
        else:
            logger.info(f"Carla server continues to run, elasped {time.time() - self._start_time} seconds")

        if self._shuffle_task:
            self._task_idx = np.random.choice(self.num_tasks)
            self._task = self._all_tasks[self._task_idx].copy()
        self.clean()

        self._wt_handler.reset(self._task["weather"])
        logger.debug("_wt_handler reset done!!")

        ev_spawn_locations = self._ev_handler.reset(self._task["ego_vehicles"])
        logger.debug("_ev_handler reset done!!")

        self._sa_handler.reset(self._task["scenario_actors"], self._ev_handler.ego_vehicles)
        logger.debug("_sa_handler reset done!!")

        self._zw_handler.reset(self._task["num_zombie_walkers"], ev_spawn_locations)
        logger.debug("_zw_handler reset done!!")

        self._zv_handler.reset(self._task["num_zombie_vehicles"], ev_spawn_locations)
        logger.debug("_zv_handler reset done!!")

        self._om_handler.reset(self._ev_handler.ego_vehicles)
        logger.debug("_om_handler reset done!!")

        self._world.tick()

        snap_shot = self._world.get_snapshot()
        self._timestamp = {
            "step": 0,
            "frame": snap_shot.timestamp.frame,
            "relative_wall_time": 0.0,
            "wall_time": snap_shot.timestamp.platform_timestamp,
            "relative_simulation_time": 0.0,
            "simulation_time": snap_shot.timestamp.elapsed_seconds,
            "start_frame": snap_shot.timestamp.frame,
            "start_wall_time": snap_shot.timestamp.platform_timestamp,
            "start_simulation_time": snap_shot.timestamp.elapsed_seconds,
        }

        _, _, _ = self._ev_handler.tick(self.timestamp)
        # get obeservations
        obs_dict = self._om_handler.get_observation(self.timestamp)
        return obs_dict

    def step(self, control_dict):
        self._ev_handler.apply_control(control_dict)
        self._sa_handler.tick()
        # tick world
        self._world.tick()

        # update timestamp
        snap_shot = self._world.get_snapshot()
        self._timestamp["step"] = snap_shot.timestamp.frame - self._timestamp["start_frame"]
        self._timestamp["frame"] = snap_shot.timestamp.frame
        self._timestamp["wall_time"] = snap_shot.timestamp.platform_timestamp
        self._timestamp["relative_wall_time"] = self._timestamp["wall_time"] - self._timestamp["start_wall_time"]
        self._timestamp["simulation_time"] = snap_shot.timestamp.elapsed_seconds
        self._timestamp["relative_simulation_time"] = (
            self._timestamp["simulation_time"] - self._timestamp["start_simulation_time"]
        )

        reward_dict, done_dict, info_dict = self._ev_handler.tick(self.timestamp)

        # get observations
        obs_dict = self._om_handler.get_observation(self.timestamp)

        # update weather
        self._wt_handler.tick(snap_shot.timestamp.delta_seconds)

        if self._timestamp["step"] % 200 == 0:
            Image.fromarray(obs_dict["hero"]["birdview"]["rendered"]).save(f"birdview_{self._carla_map}.png")
            completed = info_dict["hero"]["route_completion"]["route_completed_in_m"]
            length = info_dict["hero"]["route_completion"]["route_length_in_m"]
            print(
                f"[step={self._timestamp['step']}, {self._carla_map}] completed: {completed:.2f}m, length: {length:.2f}m, "
                f"ratio: {completed / length * 100:.2f}%"
            )

        # num_walkers = len(self._world.get_actors().filter("*walker.pedestrian*"))
        # num_vehicles = len(self._world.get_actors().filter("vehicle*"))
        # logger.debug(f"num_walkers: {num_walkers}, num_vehicles: {num_vehicles}, ")

        return obs_dict, reward_dict, done_dict, done_dict, info_dict

    def _init_client(self, carla_map, host, port, seed=2021, no_rendering=False):
        client, world, tm = None, None, None
        server_retry_count = 0
        max_retries_before_restart = 3
        total_retry = 0
        max_total_retries = 12

        while tm is None:
            if total_retry >= max_total_retries:
                raise RuntimeError(f"Failed to connect to Carla server after max retries: {max_total_retries}")
            try:
                client = carla.Client(host, port)
                client.set_timeout(60.0)
                world = client.load_world(carla_map)
                tm = client.get_trafficmanager(port + 6000)
                server_retry_count = 0  # reset on success
            except RuntimeError as re:
                total_retry += 1
                server_retry_count += 1
                if "timeout" not in str(re) and "time-out" not in str(re):
                    logger.info(f"Could not connect to Carla server because: {re}")
                client = None
                world = None
                tm = None  # back to retry

                if server_retry_count >= max_retries_before_restart:
                    logger.info(
                        f"Restarting Carla server due to repeated connection errors: {server_retry_count}/{max_retries_before_restart}"
                    )
                    # Restart the Carla server after repeated connection errors
                    self._server_manager.restart()
                    server_retry_count = 0
                logger.info(f"Waiting for 5 seconds before retrying... {total_retry}/{max_total_retries}")
                time.sleep(5)

        self._client = client
        self._world = world
        self._tm = tm

        self.set_sync_mode(True)
        self.set_no_rendering_mode(self._world, no_rendering)
        # self._tm.set_hybrid_physics_mode(True)
        # self._tm.set_global_distance_to_leading_vehicle(5.0)

        set_random_seed(self._seed, using_cuda=True)
        self._tm.set_random_device_seed(self._seed)

        self._world.tick()

    def set_sync_mode(self, sync):
        settings = self._world.get_settings()
        settings.synchronous_mode = sync
        settings.fixed_delta_seconds = 0.05
        settings.deterministic_ragdolls = True
        self._world.apply_settings(settings)
        self._tm.set_synchronous_mode(sync)

    @staticmethod
    def set_no_rendering_mode(world, no_rendering):
        settings = world.get_settings()
        settings.no_rendering_mode = no_rendering
        world.apply_settings(settings)

    @property
    def timestamp(self):
        if not hasattr(self, "_timestamp"):
            raise ValueError("timestamp is not set. Please call reset() first.")
        return self._timestamp.copy()

    @timestamp.setter
    def timestamp(self, value):
        self._timestamp = value

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()
        logger.debug("env __exit__!")

    def close(self):
        self.clean()
        self.set_sync_mode(False)
        self._client = None
        self._world = None
        if self._tm is not None:
            self._tm.shut_down()
            self._tm = None

    def clean(self):
        self._sa_handler.clean()
        self._zw_handler.clean()
        self._zv_handler.clean()
        self._om_handler.clean()
        self._ev_handler.clean()
        self._wt_handler.clean()
        self._world.tick()
