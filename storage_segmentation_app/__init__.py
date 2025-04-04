"""
Storage Segmentation App
========================

A Python application for detecting and segmenting storage furniture and compartments.

This package provides tools for:
- Detecting storage furniture in images
- Segmenting individual compartments within storage units
- Visualizing the detected storage hierarchy
- Exporting structured data about the storage units

Main Components:
- StorageDetector: Class for detecting storage units and compartments
- StorageUnit: Class representing a storage unit with multiple compartments
- StorageCompartment: Class representing a compartment within a storage unit
- Visualization utilities: Functions for visualizing detection results
"""

__version__ = "0.1.0"

from .models import StorageUnit, StorageCompartment, BoundingBox
from .detection import StorageDetector
from .utils import (
    visualize_segmentation,
    create_hierarchy_tree,
    crop_image,
    resize_with_aspect_ratio,
    export_results_to_json
)

__all__ = [
    "StorageUnit",
    "StorageCompartment",
    "BoundingBox",
    "StorageDetector",
    "visualize_segmentation",
    "create_hierarchy_tree",
    "crop_image",
    "resize_with_aspect_ratio",
    "export_results_to_json"
]