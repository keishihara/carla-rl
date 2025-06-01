import argparse
import time

import gymnasium as gym

import carla_env  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-name", type=str, default="Town01")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("-v", "--version", type=str, default="2")
    args = parser.parse_args()

    try:
        env = gym.make(
            f"Endless-v{args.version}",
            map_name=args.map_name,
            gpu_id=args.gpu_id,
            nullrhi=args.cpu,
            seed=args.seed,
            render_mode="rgb_array",
        )
        obs, info = env.reset()

        step = 0
        start_time = time.time()
        while True:
            # rendered = env.render()
            # Image.fromarray(rendered["hero"]["birdview"]["rendered"]).save("rendered.png")
            action = env.action_space.sample()
            action["hero"][2] = 0.0  # no brake to avoid stationary
            obs, reward_d, terminated_d, truncated_d, info_d = env.step(action)
            step += 1
            steps_per_second = step / (time.time() - start_time + 1e-6)
            print(f"Step {step}: reward: {reward_d['hero']}, speed: {steps_per_second:.2f} steps/s")

            if terminated_d["hero"] or truncated_d["hero"]:
                print(f"Terminated: {terminated_d['hero']}, Truncated: {truncated_d['hero']}")
                break

    except Exception as err:
        print(err)
        import traceback

        traceback.print_exc()
    finally:
        env.close()


if __name__ == "__main__":
    main()
