import os
import numpy as np
from pathlib import Path
from models import StorageUnit
from detectors.base_detector import BaseDetector
from detectors.yolo_nas_detector import YOLONASDetector
from detectors.grounding_dino_detector import GroundingDINODetector
from detectors.sam_detector import SAM21Detector
from config import DEFAULT_OBJECT_SIZE

# Import refactored modules
from detectors.hybrid_detector_utils import has_significant_overlap, non_maximum_suppression, transform_coordinates_back
from detectors.hybrid_detector_augmentation import apply_test_time_augmentation
from detectors.hybrid_detector_segmentation import refine_unit_with_sam
from detectors.hybrid_detector_compartments import detect_compartments as detect_unit_compartments

class HybridDetector(BaseDetector):
    """
    Class for detecting storage units and their compartments in images
    using a hybrid approach that combines multiple models:
    - YOLO-NAS for initial furniture unit detection
    - Grounding DINO for component classification
    - SAM 2.1 for precise segmentation
    """
    
    def __init__(self, confidence_threshold=0.5):
        """
        Initialize the hybrid detector with multiple models and parameters.
        
        Args:
            confidence_threshold (float): Minimum confidence score for detections
        """
        super().__init__(confidence_threshold)
        
        # Initialize the component models with slightly lower thresholds
        # to allow for ensemble decision making
        component_threshold = max(0.3, confidence_threshold - 0.2)
        
        # Initialize the component detectors
        self.unit_detector = YOLONASDetector(confidence_threshold=component_threshold)
        self.component_classifier = GroundingDINODetector(confidence_threshold=component_threshold)
        self.segmenter = SAM21Detector(confidence_threshold=component_threshold)
        
        # Define ensemble weights for each model
        self.ensemble_weights = {
            "yolo_nas": 0.4,
            "grounding_dino": 0.3,
            "sam21": 0.3
        }
        
        # Flag to enable test-time augmentation
        self.use_test_time_augmentation = True
        
        # Flag to enable uncertainty-aware ensemble
        self.use_uncertainty_aware_ensemble = True
        
        # Dynamic confidence thresholds based on furniture type
        self.dynamic_thresholds = {
            "Drawer": 0.65,      # Drawers need higher confidence due to similar appearance
            "Shelf": 0.55,       # Shelves are usually easier to detect
            "Cabinet": 0.60,     # Cabinets can be complex
            "Wardrobe": 0.70,    # Wardrobes are large but can be confused with other furniture
            "Storage Box": 0.65, # Boxes can be confused with other objects
            "Chest": 0.65,       # Chests can be confused with other furniture
            "Sideboard": 0.70,   # Sideboards can be complex
            "Dresser": 0.70,     # Dressers can be complex
            "Box": 0.65,         # Boxes can be confused with other objects
            "Basket": 0.65,      # Baskets can be confused with other objects
            "Refrigerator": 0.75, # Refrigerators are distinctive
            "Chest of Drawers": 0.70  # Complex furniture
        }
        
        print("Initialized Hybrid Detector with component models:")
        print(f"- Unit Detector: YOLO-NAS (weight: {self.ensemble_weights['yolo_nas']})")
        print(f"- Component Classifier: Grounding DINO (weight: {self.ensemble_weights['grounding_dino']})")
        print(f"- Segmenter: SAM 2.1 (weight: {self.ensemble_weights['sam21']})")
    
    def _load_models(self):
        """Load models is handled in __init__ by initializing component detectors."""
        pass
    
    def _get_dynamic_threshold(self, class_name):
        """
        Get dynamic confidence threshold based on furniture type.
        
        Args:
            class_name (str): Class name of the detected object
            
        Returns:
            float: Dynamic confidence threshold
        """
        # Default to the base confidence threshold if not specified
        return self.dynamic_thresholds.get(class_name, self.confidence_threshold)
    
    def process_image(self, image, detect_compartments=True, filter_small_segments=False, min_segment_width=None):
        """
        Process an image using the hybrid detection approach.
        
        Args:
            image (numpy.ndarray): Input image
            detect_compartments (bool): Whether to detect compartments within units
            filter_small_segments (bool): Whether to filter out small segments
            min_segment_width (int): Minimum width for segments to be included (if None, uses DEFAULT_OBJECT_SIZE% of image width)
            
        Returns:
            list: List of StorageUnit objects with detected compartments
        """
        # Store original image dimensions
        original_h, original_w = image.shape[:2]
        
        # If min_segment_width is not provided, calculate it based on DEFAULT_OBJECT_SIZE
        if filter_small_segments and min_segment_width is None:
            min_segment_width = int(original_w * (DEFAULT_OBJECT_SIZE / 100))
        
        # Apply test-time augmentation if enabled
        augmented_images = apply_test_time_augmentation(image, self.use_test_time_augmentation)
        
        # Initialize storage units list
        all_storage_units = []
        
        # Process each augmented image
        for aug_idx, aug_image in enumerate(augmented_images):
            print(f"Processing augmentation {aug_idx+1}/{len(augmented_images)}")
            
            # Step 1: Detect storage units with YOLO-NAS
            yolo_units = self.unit_detector.process_image(
                aug_image, 
                detect_compartments=False,  # We'll handle compartments separately
                filter_small_segments=filter_small_segments,
                min_segment_width=min_segment_width
            )
            
            # Step 2: Use Grounding DINO for additional unit detection and classification
            dino_units = self.component_classifier.process_image(
                aug_image,
                detect_compartments=False,  # We'll handle compartments separately
                filter_small_segments=filter_small_segments,
                min_segment_width=min_segment_width
            )
            
            # Step 3: Use SAM 2.1 for precise segmentation
            sam_units = self.segmenter.process_image(
                aug_image,
                detect_compartments=False,  # We'll handle compartments separately
                filter_small_segments=filter_small_segments,
                min_segment_width=min_segment_width
            )
            
            # Combine units from all detectors
            combined_units = []
            combined_units.extend(yolo_units)
            
            # Add DINO units that don't overlap significantly with YOLO units
            for dino_unit in dino_units:
                if not has_significant_overlap(dino_unit, combined_units):
                    combined_units.append(dino_unit)
            
            # Add SAM units that don't overlap significantly with existing units
            for sam_unit in sam_units:
                if not has_significant_overlap(sam_unit, combined_units):
                    combined_units.append(sam_unit)
            
            # For each unit, refine its segmentation using SAM
            refined_units = []
            for unit in combined_units:
                # Get dynamic threshold for this class
                dynamic_threshold = self._get_dynamic_threshold(unit.class_name)
                
                # Skip if confidence is below dynamic threshold
                if unit.confidence < dynamic_threshold:
                    print(f"Skipping {unit.class_name} with confidence {unit.confidence:.2f} (below dynamic threshold {dynamic_threshold:.2f})")
                    continue
                
                # Refine the unit's mask using SAM
                refined_unit = refine_unit_with_sam(aug_image, unit, self.segmenter)
                refined_units.append(refined_unit)
            
            # If this is not the original image (i.e., it's an augmented version),
            # transform the coordinates back to the original image space
            if aug_idx > 0:
                for unit in refined_units:
                    transform_coordinates_back(unit, aug_idx, original_w, original_h)
            
            # Add to the overall list
            all_storage_units.extend(refined_units)
        
        # Perform non-maximum suppression to remove duplicates
        final_units = non_maximum_suppression(
            all_storage_units, 
            iou_threshold=0.5, 
            use_uncertainty_aware_ensemble=self.use_uncertainty_aware_ensemble
        )
        
        # Detect compartments if requested
        if detect_compartments:
            for unit in final_units:
                detect_unit_compartments(
                    image, 
                    unit, 
                    self.component_classifier, 
                    self.segmenter,
                    filter_small_segments, 
                    min_segment_width,
                    DEFAULT_OBJECT_SIZE,
                    self.use_uncertainty_aware_ensemble
                )
        
        return final_units
    
    def _get_all_segments(self, image):
        """
        Get all segments from an image using SAM.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            list: List of StorageUnit objects representing segments
        """
        return self.segmenter.process_image_all_segments(image)