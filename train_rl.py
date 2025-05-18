import logging
import multiprocessing
from pathlib import Path

import gymnasium as gym
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import server_utils
from agents.rl_birdview.utils.wandb_callback import WandbCallback
from carla_gym.utils import config_utils

multiprocessing.set_start_method("spawn", force=True)  # we might not need this
logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="train_rl_local")
def main(cfg: DictConfig):
    if cfg.kill_running:
        server_utils.kill_carla()
    set_random_seed(cfg.seed, using_cuda=True)

    initial_port = 2000
    streaming_port = 0
    train_envs = cfg.train_envs

    env_configs = []
    port = initial_port
    seed = cfg.seed
    cont = 0
    for train_env in train_envs:
        for gpu in train_env["gpu"]:
            single_env_cfg = OmegaConf.to_container(train_env)
            single_env_cfg["gpu"] = gpu
            single_env_cfg["port"] = port
            single_env_cfg["streaming_port"] = streaming_port
            single_env_cfg["seed"] = seed
            env_configs.append(single_env_cfg)
            port += 1000
            seed += 1
            cont += 1
    print(port)
    print(cont)

    try:
        agent_name = cfg.actors[cfg.ev_id].agent

        last_checkpoint_path = Path(hydra.utils.get_original_cwd()) / "outputs" / "checkpoint.txt"
        if last_checkpoint_path.exists():
            with open(last_checkpoint_path) as f:
                wb_run_path = f.read().strip()
                if wb_run_path:
                    cfg.agent[agent_name].wb_run_path = wb_run_path

        OmegaConf.save(config=cfg.agent[agent_name], f="config_agent.yaml")

        AgentClass = config_utils.load_entry_point(cfg.agent[agent_name].entry_point)
        agent = AgentClass("config_agent.yaml")

        cfg_agent = OmegaConf.load("config_agent.yaml")
        obs_configs = {cfg.ev_id: OmegaConf.to_container(cfg_agent.obs_configs)}
        reward_configs = {cfg.ev_id: OmegaConf.to_container(cfg.actors[cfg.ev_id].reward)}
        terminal_configs = {cfg.ev_id: OmegaConf.to_container(cfg.actors[cfg.ev_id].terminal)}

        EnvWrapper = config_utils.load_entry_point(cfg_agent.env_wrapper.entry_point)
        wrapper_kargs = cfg_agent.env_wrapper.kwargs

        config_utils.check_h5_maps(cfg.train_envs, obs_configs)

        def env_maker(config):
            # launch carla server inside subprocess
            server_manager = server_utils.CarlaServerManager(
                rpc_port=config["port"],
                streaming_port=config["streaming_port"],
                server_life_time_sec=40 * 60,
                use_apptainer=True,
            )
            server_manager.start(gpu_index=config["gpu"])

            logger.info(f"making port {config['port']}, env_id={config['env_id']}")
            env = gym.make(
                config["env_id"],
                obs_configs=obs_configs,
                reward_configs=reward_configs,
                terminal_configs=terminal_configs,
                host="localhost",
                port=config["port"],
                seed=config["seed"],
                no_rendering=True,
                server_manager=server_manager,
                render_mode="rgb_array",
                **config["env_configs"],
            )
            env = EnvWrapper(env, **wrapper_kargs)
            return env

        if cfg.dummy or len(env_configs) == 1:
            env = DummyVecEnv([lambda config=config: env_maker(config) for config in env_configs])
        else:
            env = SubprocVecEnv([lambda config=config: env_maker(config) for config in env_configs])

        wb_callback = WandbCallback(cfg, env)
        callback = CallbackList([wb_callback])
        env.reset()

        with open(last_checkpoint_path, "w") as f:
            f.write(wandb.run.path)

        agent.learn(env, total_timesteps=int(cfg.total_timesteps), callback=callback, seed=cfg.seed)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise e

    finally:
        pass


if __name__ == "__main__":
    main()
    logger.info("train_rl.py DONE!")
