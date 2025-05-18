import argparse
import time
from pathlib import Path

import carla
from tqdm import tqdm

BASE_DIR = Path("logs")
RGB_DIR = BASE_DIR / "rgb"
DEPTH_DIR = BASE_DIR / "depth"


def setup_directories():
    RGB_DIR.mkdir(parents=True, exist_ok=True)
    DEPTH_DIR.mkdir(parents=True, exist_ok=True)


def spawn_vehicle(world):
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("vehicle.*")[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    return vehicle


def setup_camera(world, vehicle, camera_type, output_dir, transform):
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find(camera_type)
    camera_bp.set_attribute("image_size_x", "800")
    camera_bp.set_attribute("image_size_y", "600")
    camera_bp.set_attribute("fov", "90")
    output_dir.mkdir(parents=True, exist_ok=True)
    camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
    return camera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", type=str, default="Town15")
    args = parser.parse_args()

    client = carla.Client("localhost", args.port)
    client.set_timeout(20.0)
    available_maps = client.get_available_maps()
    print("Available maps:", available_maps)

    client.load_world(args.town)

    setup_directories()

    try:
        world = client.get_world()

        vehicle = spawn_vehicle(world)

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        rgb_camera = setup_camera(world, vehicle, "sensor.camera.rgb", RGB_DIR, camera_transform)
        depth_camera = setup_camera(world, vehicle, "sensor.camera.depth", DEPTH_DIR, camera_transform)

        capture_count = 0
        max_captures = 10

        progress_bar = tqdm(total=max_captures, desc="Capturing images", unit="frame")

        def save_rgb_image(image):
            nonlocal capture_count
            if capture_count < max_captures:
                image.save_to_disk(RGB_DIR / f"rgb_{image.frame:06d}.png")
                capture_count += 1
                progress_bar.update(1)
            if capture_count >= max_captures:
                rgb_camera.stop()

        def save_depth_image(image):
            if capture_count <= max_captures:
                image.save_to_disk(DEPTH_DIR / f"depth_{image.frame:06d}.png")

        # camera listener
        rgb_camera.listen(save_rgb_image)
        depth_camera.listen(save_depth_image)

        vehicle.set_autopilot(True)

        while capture_count < max_captures:
            time.sleep(1)

        progress_bar.close()

    finally:
        if rgb_camera:
            rgb_camera.stop()
            rgb_camera.destroy()
        if depth_camera:
            depth_camera.stop()
            depth_camera.destroy()
        if vehicle:
            vehicle.destroy()
        print("All cleaned up!")


if __name__ == "__main__":
    main()
