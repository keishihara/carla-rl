"""Event handler registry for CARLA environment components."""

from __future__ import annotations

import logging
import weakref
from collections import defaultdict
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Type aliases
Handler = Callable[..., Any]
EventType = str


class HandlerRegistry:
    """Registry for managing event handlers across CARLA environment components.

    Provides a centralized way to register, unregister, and dispatch events
    between different components of the CARLA environment system.
    """

    def __init__(self):
        """Initialize the HandlerRegistry."""
        self._handlers: dict[EventType, list[Handler]] = defaultdict(list)
        self._weak_handlers: dict[EventType, list[weakref.ref]] = defaultdict(list)
        self._handler_ids: dict[int, tuple[EventType, Handler]] = {}
        self._next_id = 0
        self._disabled_events: set[EventType] = set()

    def register(
        self,
        event_type: EventType,
        handler: Handler,
        weak_ref: bool = False,
    ) -> int:
        """Register an event handler for a specific event type.

        Args:
            event_type: The type of event to handle
            handler: The handler function to call when the event occurs
            weak_ref: If True, store only a weak reference to the handler

        Returns:
            Handler ID that can be used to unregister the handler

        Raises:
            ValueError: If handler is not callable
        """
        if not callable(handler):
            raise ValueError("Handler must be callable")

        handler_id = self._next_id
        self._next_id += 1

        if weak_ref:
            # Store weak reference to handler
            weak_handler = weakref.ref(handler, lambda ref: self._cleanup_weak_ref(event_type, ref))
            self._weak_handlers[event_type].append(weak_handler)
            self._handler_ids[handler_id] = (event_type, weak_handler)
        else:
            # Store strong reference to handler
            self._handlers[event_type].append(handler)
            self._handler_ids[handler_id] = (event_type, handler)

        logger.debug(f"Registered handler {handler_id} for event '{event_type}' (weak_ref={weak_ref})")
        return handler_id

    def unregister(self, handler_id: int) -> bool:
        """Unregister an event handler by its ID.

        Args:
            handler_id: The ID returned by register()

        Returns:
            True if handler was found and removed, False otherwise
        """
        if handler_id not in self._handler_ids:
            logger.warning(f"Handler ID {handler_id} not found")
            return False

        event_type, handler_or_ref = self._handler_ids[handler_id]

        # Remove from appropriate collection
        if isinstance(handler_or_ref, weakref.ref):
            if handler_or_ref in self._weak_handlers[event_type]:
                self._weak_handlers[event_type].remove(handler_or_ref)
        elif handler_or_ref in self._handlers[event_type]:
            self._handlers[event_type].remove(handler_or_ref)

        # Clean up handler ID mapping
        del self._handler_ids[handler_id]

        logger.debug(f"Unregistered handler {handler_id} for event '{event_type}'")
        return True

    def unregister_all(self, event_type: EventType | None = None) -> int:
        """Unregister all handlers for a specific event type or all events.

        Args:
            event_type: Event type to clear handlers for, or None for all events

        Returns:
            Number of handlers removed
        """
        removed_count = 0

        if event_type is None:
            # Remove all handlers for all events
            removed_count = len(self._handler_ids)
            self._handlers.clear()
            self._weak_handlers.clear()
            self._handler_ids.clear()
            logger.info(f"Unregistered all {removed_count} handlers")
        else:
            # Remove handlers for specific event type
            handlers_to_remove = []

            for handler_id, (stored_event_type, _) in self._handler_ids.items():
                if stored_event_type == event_type:
                    handlers_to_remove.append(handler_id)

            for handler_id in handlers_to_remove:
                self.unregister(handler_id)
                removed_count += 1

            # Clean up empty collections
            if event_type in self._handlers and not self._handlers[event_type]:
                del self._handlers[event_type]
            if event_type in self._weak_handlers and not self._weak_handlers[event_type]:
                del self._weak_handlers[event_type]

            logger.info(f"Unregistered {removed_count} handlers for event '{event_type}'")

        return removed_count

    def dispatch(
        self,
        event_type: EventType,
        *args: Any,
        **kwargs: Any,
    ) -> list[Any]:
        """Dispatch an event to all registered handlers.

        Args:
            event_type: The type of event to dispatch
            *args: Positional arguments to pass to handlers
            **kwargs: Keyword arguments to pass to handlers

        Returns:
            List of return values from all handlers
        """
        if event_type in self._disabled_events:
            logger.debug(f"Event '{event_type}' is disabled, skipping dispatch")
            return []

        results: list[Any] = []
        handlers_called = 0

        # Call strong reference handlers
        for handler in self._handlers.get(event_type, []):
            try:
                result = handler(*args, **kwargs)
                results.append(result)
                handlers_called += 1
            except Exception as e:
                logger.error(f"Handler error for event '{event_type}': {e}")

        # Call weak reference handlers (clean up dead references)
        weak_handlers = self._weak_handlers.get(event_type, [])
        live_handlers = []

        for weak_handler in weak_handlers:
            handler = weak_handler()
            if handler is not None:
                try:
                    result = handler(*args, **kwargs)
                    results.append(result)
                    handlers_called += 1
                    live_handlers.append(weak_handler)
                except Exception as e:
                    logger.error(f"Weak handler error for event '{event_type}': {e}")
                    live_handlers.append(weak_handler)

        # Update weak handlers list to remove dead references
        if len(live_handlers) != len(weak_handlers):
            self._weak_handlers[event_type] = live_handlers

        logger.debug(f"Dispatched event '{event_type}' to {handlers_called} handlers")
        return results

    def enable_event(self, event_type: EventType) -> None:
        """Enable dispatching for a specific event type.

        Args:
            event_type: Event type to enable
        """
        if event_type in self._disabled_events:
            self._disabled_events.remove(event_type)
            logger.info(f"Enabled event '{event_type}'")

    def disable_event(self, event_type: EventType) -> None:
        """Disable dispatching for a specific event type.

        Args:
            event_type: Event type to disable
        """
        self._disabled_events.add(event_type)
        logger.info(f"Disabled event '{event_type}'")

    def is_event_enabled(self, event_type: EventType) -> bool:
        """Check if an event type is enabled for dispatching.

        Args:
            event_type: Event type to check

        Returns:
            True if event is enabled, False if disabled
        """
        return event_type not in self._disabled_events

    def get_handler_count(self, event_type: EventType | None = None) -> int:
        """Get the number of registered handlers.

        Args:
            event_type: Event type to count handlers for, or None for total

        Returns:
            Number of registered handlers
        """
        if event_type is None:
            return len(self._handler_ids)

        count = len(self._handlers.get(event_type, []))
        count += len([ref for ref in self._weak_handlers.get(event_type, []) if ref() is not None])
        return count

    def get_event_types(self) -> set[EventType]:
        """Get all registered event types.

        Returns:
            Set of event types that have at least one handler
        """
        event_types = set(self._handlers.keys())
        event_types.update(self._weak_handlers.keys())
        return event_types

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics.

        Returns:
            Dictionary containing registry statistics
        """
        total_handlers = len(self._handler_ids)
        strong_handlers = sum(len(handlers) for handlers in self._handlers.values())
        weak_handlers = sum(
            len([ref for ref in handlers if ref() is not None]) for handlers in self._weak_handlers.values()
        )

        return {
            "total_handlers": total_handlers,
            "strong_handlers": strong_handlers,
            "weak_handlers": weak_handlers,
            "event_types": len(self.get_event_types()),
            "disabled_events": len(self._disabled_events),
            "next_handler_id": self._next_id,
        }

    def _cleanup_weak_ref(self, event_type: EventType, dead_ref: weakref.ref) -> None:
        """Clean up a dead weak reference.

        Args:
            event_type: Event type the reference was registered for
            dead_ref: The dead weak reference to clean up
        """
        if event_type in self._weak_handlers and dead_ref in self._weak_handlers[event_type]:
            self._weak_handlers[event_type].remove(dead_ref)
            logger.debug(f"Cleaned up dead weak reference for event '{event_type}'")


# Global instance for convenience
_global_registry: HandlerRegistry | None = None


def get_global_registry() -> HandlerRegistry:
    """Get the global HandlerRegistry instance, creating it if necessary.

    Returns:
        The global HandlerRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = HandlerRegistry()
    return _global_registry


def reset_global_registry() -> None:
    """Reset the global HandlerRegistry instance."""
    global _global_registry
    if _global_registry is not None:
        _global_registry.unregister_all()
    _global_registry = HandlerRegistry()
