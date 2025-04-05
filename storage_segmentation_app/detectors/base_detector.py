import os
import numpy as np
import cv2
from abc import ABC, abstractmethod
from pathlib import Path
from models import StorageUnit, StorageCompartment
from config import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_OBJECT_SIZE

class BaseDetector(ABC):
    """
    Abstract base class for storage unit and compartment detection.
    """
    
    def __init__(self, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
        """
        Initialize the detector with parameters.
        
        Args:
            confidence_threshold (float): Minimum confidence score for detections
        """
        self.confidence_threshold = confidence_threshold
        self.input_size = 640
        
        # Define storage furniture classes with hierarchical structure
        # Format: {class_id: (class_name, parent_class_id, is_unit, is_compartment)}
        # parent_class_id of None means it's a top-level class
        # is_unit and is_compartment flags indicate if the class can be a unit, compartment, or both
        self.hierarchical_classes = {
            # Storage Units (Furniture)
            200: ("Storage Unit", None, True, False),  # Generic parent class
            
            # Large Storage Furniture
            201: ("Large Storage", 200, True, False),  # Parent for large storage furniture
            102: ("Cabinet", 201, True, True),
            103: ("Wardrobe", 201, True, False),
            106: ("Sideboard", 201, True, False),
            109: ("Dresser", 201, True, False),
            110: ("Hutch", 201, True, False),
            111: ("Credenza", 201, True, False),
            125: ("Refrigerator", 201, True, False),
            126: ("Chest of Drawers", 201, True, False),
            127: ("Bookcase", 201, True, False),
            128: ("Display Cabinet", 201, True, False),
            129: ("Entertainment Center", 201, True, False),
            130: ("Armoire", 201, True, False),
            
            # Medium Storage Furniture
            202: ("Medium Storage", 200, True, False),  # Parent for medium storage furniture
            105: ("Chest", 202, True, False),
            107: ("Console Table", 202, True, False),
            108: ("Nightstand", 202, True, False),
            131: ("Side Table", 202, True, False),
            132: ("Coffee Table", 202, True, False),
            133: ("TV Stand", 202, True, False),
            
            # Small Storage Items
            203: ("Small Storage", 200, True, True),  # Parent for small storage items
            104: ("Storage Box", 203, True, True),
            112: ("Cubby", 203, True, True),
            113: ("Bin", 203, True, True),
            116: ("Basket", 203, True, True),
            118: ("Box", 203, True, True),
            119: ("Container", 203, True, True),
            134: ("Jewelry Box", 203, True, True),
            135: ("Decorative Box", 203, True, True),
            
            # Storage Components
            204: ("Storage Component", None, False, True),  # Generic parent for components
            
            # Horizontal Components
            205: ("Horizontal Component", 204, False, True),  # Parent for horizontal components
            101: ("Shelf", 205, False, True),
            117: ("Tray", 205, False, True),
            136: ("Platform", 205, False, True),
            137: ("Surface", 205, False, True),
            
            # Vertical Components
            206: ("Vertical Component", 204, False, True),  # Parent for vertical components
            120: ("Divider", 206, False, True),
            138: ("Panel", 206, False, True),
            139: ("Partition", 206, False, True),
            
            # Movable Components
            207: ("Movable Component", 204, False, True),  # Parent for movable components
            100: ("Drawer", 207, False, True),
            115: ("Cabinet Door", 207, False, True),
            140: ("Sliding Door", 207, False, True),
            141: ("Hinged Door", 207, False, True),
            142: ("Pull-out Tray", 207, False, True),
            
            # Organizational Components
            208: ("Organizational Component", 204, False, True),  # Parent for organizational components
            114: ("Compartment", 208, False, True),
            121: ("Rack", 208, False, True),
            122: ("Hanger", 208, False, True),
            123: ("Hook", 208, False, True),
            124: ("Organizer", 208, False, True),
            143: ("Divider Insert", 208, False, True),
            144: ("Shelf Insert", 208, False, True),
            
            # COCO classes that map to storage furniture
            75: ("Vase", 203, True, True),        # vase (COCO class)
            73: ("Bookshelf", 201, True, False)   # book (COCO class, will be treated as bookshelf)
        }
        
        # Create flat dictionaries for backward compatibility
        self.storage_furniture_classes = {}
        self.unit_classes = {}
        self.compartment_classes = {}
        
        # Populate the flat dictionaries from the hierarchical structure
        for class_id, (class_name, parent_id, is_unit, is_compartment) in self.hierarchical_classes.items():
            if is_unit or is_compartment:
                self.storage_furniture_classes[class_id] = class_name
            
            if is_unit:
                self.unit_classes[class_id] = class_name
            
            if is_compartment:
                self.compartment_classes[class_id] = class_name
        
        # Keep the full list of storage-specific classes for reference
        self.storage_specific_classes = {class_id: class_info[0] for class_id, class_info in self.hierarchical_classes.items()}
        
        # Multi-label classification support
        self.multi_label_enabled = False
        self.multi_label_threshold = 0.3  # Threshold for secondary labels
    
    def get_parent_class(self, class_id):
        """
        Get the parent class of a given class ID.
        
        Args:
            class_id (int): Class ID to get parent for
            
        Returns:
            tuple: (parent_class_id, parent_class_name) or (None, None) if no parent
        """
        if class_id in self.hierarchical_classes:
            parent_id = self.hierarchical_classes[class_id][1]
            if parent_id is not None:
                parent_name = self.hierarchical_classes[parent_id][0]
                return parent_id, parent_name
        
        return None, None
    
    def get_class_hierarchy(self, class_id):
        """
        Get the full hierarchy path for a class ID.
        
        Args:
            class_id (int): Class ID to get hierarchy for
            
        Returns:
            list: List of (class_id, class_name) tuples from root to the given class
        """
        if class_id not in self.hierarchical_classes:
            return []
        
        hierarchy = [(class_id, self.hierarchical_classes[class_id][0])]
        parent_id = self.hierarchical_classes[class_id][1]
        
        while parent_id is not None:
            hierarchy.insert(0, (parent_id, self.hierarchical_classes[parent_id][0]))
            parent_id = self.hierarchical_classes[parent_id][1]
        
        return hierarchy
    
    def is_unit_class(self, class_id):
        """Check if a class ID is a storage unit class."""
        if class_id in self.hierarchical_classes:
            return self.hierarchical_classes[class_id][2]
        return False
    
    def is_compartment_class(self, class_id):
        """Check if a class ID is a storage compartment class."""
        if class_id in self.hierarchical_classes:
            return self.hierarchical_classes[class_id][3]
        return False
    
    def enable_multi_label_classification(self, enabled=True, threshold=0.3):
        """
        Enable or disable multi-label classification.
        
        Args:
            enabled (bool): Whether to enable multi-label classification
            threshold (float): Confidence threshold for secondary labels
        """
        self.multi_label_enabled = enabled
        self.multi_label_threshold = threshold
    
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
            min_segment_width (int): Minimum width for segments to be included (if None, uses DEFAULT_OBJECT_SIZE% of image width)
            
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
    
    def process_image_all_segments(self, image):
        """
        Process an image to show ALL segmentation possible by the model without any filtering.
        This method completely disregards the concepts of "compartments" and "units" and
        displays the raw segmentation output from the model.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            list: List of StorageUnit objects representing raw segments (not actual storage units)
        """
        # This is a default implementation that should be overridden by each detector
        # to provide model-specific raw segmentation output
        return self._get_all_segments(image)
    
    @abstractmethod
    def _get_all_segments(self, image):
        """
        Get all possible segments from the model without any filtering.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            list: List of StorageUnit objects representing raw segments
        """
        pass
