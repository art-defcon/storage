import os
import numpy as np
import cv2
from abc import ABC, abstractmethod
from pathlib import Path
from models import StorageUnit, StorageCompartment

class BaseDetector(ABC):
    """
    Abstract base class for storage unit and compartment detection.
    """
    
    def __init__(self, confidence_threshold=0.5):
        """
        Initialize the detector with parameters.
        
        Args:
            confidence_threshold (float): Minimum confidence score for detections
        """
        self.confidence_threshold = confidence_threshold
        self.input_size = 640
        
        # Define storage furniture classes - ONLY include storage furniture items
        self.storage_furniture_classes = {
            # COCO classes that map to storage furniture
            75: "Vase",        # vase (COCO class)
            73: "Bookshelf",   # book (COCO class, will be treated as bookshelf)
            
            # Storage-specific classes
            100: "Drawer",
            101: "Shelf",
            102: "Cabinet",
            103: "Wardrobe",
            104: "Storage Box",
            105: "Chest",
            106: "Sideboard",
            109: "Dresser",
            118: "Box",
            116: "Basket",
            125: "Refrigerator",
            126: "Chest of Drawers"
        }
        
        # Use the storage furniture classes for both unit and compartment detection
        self.unit_classes = self.storage_furniture_classes.copy()
        self.compartment_classes = self.storage_furniture_classes.copy()
        
        # Keep the full list of storage-specific classes for reference
        # but we'll only use the ones in storage_furniture_classes for detection
        self.storage_specific_classes = {
            100: "Drawer",
            101: "Shelf",
            102: "Cabinet",
            103: "Wardrobe",
            104: "Storage Box",
            105: "Chest",
            106: "Sideboard",
            107: "Console Table",
            108: "Nightstand",
            109: "Dresser",
            110: "Hutch",
            111: "Credenza",
            112: "Cubby",
            113: "Bin",
            114: "Compartment",
            115: "Cabinet Door",
            116: "Basket",
            117: "Tray",
            118: "Box",
            119: "Container",
            120: "Divider",
            121: "Rack",
            122: "Hanger",
            123: "Hook",
            124: "Organizer",
            125: "Refrigerator",
            126: "Chest of Drawers"
        }
    
    @abstractmethod
    def _load_models(self):
        """Load the detection models."""
        pass
    
    def _resize_image(self, image):
        """
        Resize image to the model's input size while preserving aspect ratio.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            tuple: (resized_image, scale_x, scale_y)
        """
        h, w = image.shape[:2]
        
        # Calculate scale factors
        scale_x = w / self.input_size
        scale_y = h / self.input_size
        
        # Resize image
        resized_image = cv2.resize(image, (self.input_size, self.input_size))
        
        return resized_image, scale_x, scale_y
    
    def _scale_coordinates(self, x1, y1, x2, y2, scale_x, scale_y):
        """
        Scale coordinates back to original image size.
        
        Args:
            x1, y1, x2, y2: Bounding box coordinates
            scale_x, scale_y: Scale factors
            
        Returns:
            tuple: Scaled coordinates (x1, y1, x2, y2)
        """
        return (
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y)
        )
    
    @abstractmethod
    def process_image(self, image, detect_compartments=True, filter_small_segments=False, min_segment_width=None):
        """
        Process an image to detect storage units and their compartments.
        
        Args:
            image (numpy.ndarray): Input image
            detect_compartments (bool): Whether to detect compartments within units
            filter_small_segments (bool): Whether to filter out small segments
            min_segment_width (int): Minimum width for segments to be included (if None, uses 15% of image width)
            
        Returns:
            list: List of StorageUnit objects with detected compartments
        """
        pass
    
    @abstractmethod
    def _detect_compartments(self, image, storage_unit, filter_small_segments=False, min_segment_width=None):
        """
        Detect compartments within a storage unit.
        
        Args:
            image (numpy.ndarray): Input image
            storage_unit (StorageUnit): Storage unit to detect compartments in
            filter_small_segments (bool): Whether to filter out small segments
            min_segment_width (int): Minimum width for segments to be included
        """
        pass
