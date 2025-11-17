"""
Tracking module: Multi-object tracking with threat prioritization.

Public API:
    - TrackingController: Main interface for tracking system
    - Track: Individual track state (for advanced usage)
"""
from tracking.controller import TrackingController
from tracking.track import Track
from tracking.multi_tracker import MultiObjectTracker
from tracking.matcher import HungarianMatcher
from tracking.threat_analyzer import ThreatAnalyzer

__all__ = [
    'TrackingController',
    'Track',
    'MultiObjectTracker',
    'HungarianMatcher',
    'ThreatAnalyzer',
]