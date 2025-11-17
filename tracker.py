"""
Backwards-compatible wrapper for the new modular tracking system.
This file maintains API compatibility with existing code (tracking_thread.py).

The actual implementation is now in the 'tracking' module.
"""
from tracking import TrackingController

# Export the main class with the old name for compatibility
Tracker = TrackingController