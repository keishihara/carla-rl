"""CARLA simulation time management and synchronization utilities."""

from __future__ import annotations

import logging
import time
from typing import Any

import carla

logger = logging.getLogger(__name__)


class TimeManager:
    """Manages CARLA simulation time, synchronization mode, and frame timing."""

    def __init__(
        self,
        world: carla.World,
        traffic_manager: carla.TrafficManager | None = None,
        fixed_delta_seconds: float = 0.05,
        deterministic_ragdolls: bool = True,
    ):
        """Initialize the TimeManager.

        Args:
            world: CARLA world instance
            traffic_manager: CARLA traffic manager instance
            fixed_delta_seconds: Fixed time step for synchronous mode
            deterministic_ragdolls: Whether to enable deterministic ragdoll physics
        """
        self.world = world
        self.traffic_manager = traffic_manager
        self.fixed_delta_seconds = fixed_delta_seconds
        self.deterministic_ragdolls = deterministic_ragdolls

        self._sync_mode_enabled = False
        self._original_settings: carla.WorldSettings | None = None
        self._step_count = 0
        self._simulation_start_time = 0.0
        self._real_start_time = 0.0

    def enable_sync_mode(self, fixed_delta_seconds: float | None = None) -> None:
        """Enable synchronous simulation mode.

        Args:
            fixed_delta_seconds: Override the default fixed time step
        """
        if self._sync_mode_enabled:
            logger.warning("Sync mode is already enabled")
            return

        # Store original settings for restoration
        self._original_settings = self.world.get_settings()

        # Apply new synchronous settings
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = fixed_delta_seconds or self.fixed_delta_seconds
        settings.deterministic_ragdolls = self.deterministic_ragdolls

        self.world.apply_settings(settings)

        # Configure traffic manager if available
        if self.traffic_manager is not None:
            self.traffic_manager.set_synchronous_mode(True)

        self._sync_mode_enabled = True
        self._step_count = 0
        self._simulation_start_time = self.world.get_snapshot().timestamp.elapsed_seconds
        self._real_start_time = time.time()

        logger.info(f"Sync mode enabled with fixed_delta_seconds={settings.fixed_delta_seconds}")

    def disable_sync_mode(self) -> None:
        """Disable synchronous simulation mode and restore original settings."""
        if not self._sync_mode_enabled:
            logger.warning("Sync mode is not enabled")
            return

        # Restore original settings
        if self._original_settings is not None:
            self.world.apply_settings(self._original_settings)
        else:
            # Fallback: manually disable sync mode
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

        # Configure traffic manager if available
        if self.traffic_manager is not None:
            self.traffic_manager.set_synchronous_mode(False)

        self._sync_mode_enabled = False
        self._original_settings = None

        logger.info("Sync mode disabled")

    def tick(self, timeout: float = 10.0) -> carla.WorldSnapshot:
        """Advance simulation by one time step in synchronous mode.

        Args:
            timeout: Maximum time to wait for the tick to complete

        Returns:
            World snapshot after the tick

        Raises:
            RuntimeError: If sync mode is not enabled or tick fails
        """
        if not self._sync_mode_enabled:
            raise RuntimeError("Cannot tick: sync mode is not enabled")

        try:
            snapshot = self.world.tick(timeout)
            self._step_count += 1
            return snapshot
        except RuntimeError as e:
            logger.error(f"World tick failed: {e}")
            raise

    def get_simulation_time(self) -> float:
        """Get current simulation time in seconds."""
        return self.world.get_snapshot().timestamp.elapsed_seconds

    def get_simulation_step_count(self) -> int:
        """Get the number of simulation steps taken since sync mode was enabled."""
        return self._step_count

    def get_real_time_factor(self) -> float:
        """Calculate the real-time factor (simulation time / real time).

        Returns:
            Ratio of simulation time to real time since sync mode was enabled.
            Values > 1.0 indicate faster-than-real-time simulation.
        """
        if not self._sync_mode_enabled or self._real_start_time == 0:
            return 0.0

        current_sim_time = self.get_simulation_time()
        current_real_time = time.time()

        sim_elapsed = current_sim_time - self._simulation_start_time
        real_elapsed = current_real_time - self._real_start_time

        if real_elapsed <= 0:
            return 0.0

        return sim_elapsed / real_elapsed

    def get_fps(self) -> float:
        """Calculate average frames per second since sync mode was enabled."""
        if not self._sync_mode_enabled or self._real_start_time == 0:
            return 0.0

        real_elapsed = time.time() - self._real_start_time
        if real_elapsed <= 0:
            return 0.0

        return self._step_count / real_elapsed

    def is_sync_mode_enabled(self) -> bool:
        """Check if synchronous mode is currently enabled."""
        return self._sync_mode_enabled

    def reset_timing_stats(self) -> None:
        """Reset timing statistics while keeping sync mode enabled."""
        if self._sync_mode_enabled:
            self._step_count = 0
            self._simulation_start_time = self.world.get_snapshot().timestamp.elapsed_seconds
            self._real_start_time = time.time()

    def __enter__(self) -> TimeManager:
        """Context manager entry: enable sync mode."""
        self.enable_sync_mode()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit: disable sync mode."""
        self.disable_sync_mode()


def sync_mode(sync: bool, world: carla.World, tm: carla.TrafficManager | None = None, seed: int | None = None) -> None:
    """Set synchronous mode for the CARLA world and traffic manager.

    Legacy function for compatibility. Consider using TimeManager class instead.

    Args:
        sync: Whether to enable synchronous mode
        world: The CARLA world instance
        tm: The CARLA traffic manager. If None, no action is taken for traffic manager
        seed: The seed for the random number generator
    """
    settings = world.get_settings()
    settings.synchronous_mode = sync
    settings.fixed_delta_seconds = 0.05
    settings.deterministic_ragdolls = True
    world.apply_settings(settings)

    if tm is not None:
        tm.set_synchronous_mode(sync)
        tm.set_random_device_seed(seed)
