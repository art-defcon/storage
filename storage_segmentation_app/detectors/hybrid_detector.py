import os
import numpy as np
import cv2
from pathlib import Path
from models import StorageUnit, StorageCompartment
from detectors.base_detector import BaseDetector
from detectors.yolo_nas_detector import YOLONASDetector
from detectors.grounding_dino_detector import GroundingDINODetector
from detectors.sam_detector import SAM21Detector
from config import DEFAULT_OBJECT_SIZE

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
    
    def _apply_test_time_augmentation(self, image):
        """
        Apply test-time augmentation to the image.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            list: List of augmented images
        """
        if not self.use_test_time_augmentation:
            return [image]
        
        augmented_images = [image]  # Original image
        
        # Horizontal flip
        flipped_h = cv2.flip(image, 1)
        augmented_images.append(flipped_h)
        
        # Rotate 90 degrees
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, 90, 1.0)
        rotated_90 = cv2.warpAffine(image, rotation_matrix, (w, h))
        augmented_images.append(rotated_90)
        
        # Brightness adjustment
        bright = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
        augmented_images.append(bright)
        
        # Contrast adjustment
        contrast = cv2.convertScaleAbs(image, alpha=1.3, beta=0)
        augmented_images.append(contrast)
        
        return augmented_images
    
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
        augmented_images = self._apply_test_time_augmentation(image)
        
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
                if not self._has_significant_overlap(dino_unit, combined_units):
                    combined_units.append(dino_unit)
            
            # Add SAM units that don't overlap significantly with existing units
            for sam_unit in sam_units:
                if not self._has_significant_overlap(sam_unit, combined_units):
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
                refined_unit = self._refine_unit_with_sam(aug_image, unit)
                refined_units.append(refined_unit)
            
            # If this is not the original image (i.e., it's an augmented version),
            # transform the coordinates back to the original image space
            if aug_idx > 0:
                for unit in refined_units:
                    self._transform_coordinates_back(unit, aug_idx, original_w, original_h)
            
            # Add to the overall list
            all_storage_units.extend(refined_units)
        
        # Perform non-maximum suppression to remove duplicates
        final_units = self._non_maximum_suppression(all_storage_units)
        
        # Detect compartments if requested
        if detect_compartments:
            for unit in final_units:
                self._detect_compartments(image, unit, filter_small_segments, min_segment_width)
        
        return final_units
    
    def _has_significant_overlap(self, unit, unit_list, iou_threshold=0.5):
        """
        Check if a unit has significant overlap with any unit in the list.
        
        Args:
            unit (StorageUnit): Unit to check
            unit_list (list): List of units to check against
            iou_threshold (float): IoU threshold for significant overlap
            
        Returns:
            bool: True if significant overlap exists, False otherwise
        """
        for existing_unit in unit_list:
            # Calculate intersection over union (IoU)
            intersection_x1 = max(unit.x1, existing_unit.x1)
            intersection_y1 = max(unit.y1, existing_unit.y1)
            intersection_x2 = min(unit.x2, existing_unit.x2)
            intersection_y2 = min(unit.y2, existing_unit.y2)
            
            if intersection_x1 < intersection_x2 and intersection_y1 < intersection_y2:
                intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                unit_area = unit.width * unit.height
                existing_unit_area = existing_unit.width * existing_unit.height
                union_area = unit_area + existing_unit_area - intersection_area
                
                iou = intersection_area / union_area
                
                if iou > iou_threshold:
                    return True
        
        return False
    
    def _refine_unit_with_sam(self, image, unit):
        """
        Refine a unit's segmentation using SAM.
        
        Args:
            image (numpy.ndarray): Input image
            unit (StorageUnit): Unit to refine
            
        Returns:
            StorageUnit: Refined unit
        """
        # Crop the image to the unit's bounding box with some padding
        x1, y1, x2, y2 = unit.x1, unit.y1, unit.x2, unit.y2
        
        # Add padding (10% of width/height)
        pad_x = int(unit.width * 0.1)
        pad_y = int(unit.height * 0.1)
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1_pad = max(0, x1 - pad_x)
        y1_pad = max(0, y1 - pad_y)
        x2_pad = min(w, x2 + pad_x)
        y2_pad = min(h, y2 + pad_y)
        
        # Crop the image
        cropped_image = image[y1_pad:y2_pad, x1_pad:x2_pad]
        
        # Skip if the cropped image is too small
        if cropped_image.shape[0] < 10 or cropped_image.shape[1] < 10:
            return unit
        
        # Use SAM to segment the cropped image
        sam_results = self.segmenter.process_image_all_segments(cropped_image)
        
        # If no SAM results, return the original unit
        if not sam_results:
            return unit
        
        # Find the SAM segment with the highest IoU with the unit's bounding box
        best_iou = 0
        best_mask = None
        
        # Create a simple rectangular mask for the original unit (in cropped coordinates)
        unit_mask = np.zeros((y2_pad - y1_pad, x2_pad - x1_pad), dtype=bool)
        unit_mask[y1 - y1_pad:y2 - y1_pad, x1 - x1_pad:x2 - x1_pad] = True
        
        for sam_unit in sam_results:
            if sam_unit.mask is not None:
                # Calculate IoU between the unit's rectangular mask and the SAM mask
                intersection = np.logical_and(unit_mask, sam_unit.mask).sum()
                union = np.logical_or(unit_mask, sam_unit.mask).sum()
                
                if union > 0:
                    iou = intersection / union
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_mask = sam_unit.mask
        
        # If a good mask was found, update the unit's mask
        if best_mask is not None and best_iou > 0.3:
            # Create a full-sized mask
            full_mask = np.zeros(image.shape[:2], dtype=bool)
            
            # Place the cropped mask in the correct position
            full_mask[y1_pad:y2_pad, x1_pad:x2_pad] = best_mask
            
            # Update the unit's mask
            unit.mask = full_mask
            
            # Recalculate bounding box from the mask
            y_indices, x_indices = np.where(full_mask)
            if len(y_indices) > 0 and len(x_indices) > 0:
                unit.x1 = int(np.min(x_indices))
                unit.y1 = int(np.min(y_indices))
                unit.x2 = int(np.max(x_indices))
                unit.y2 = int(np.max(y_indices))
        
        return unit
    
    def _transform_coordinates_back(self, unit, aug_idx, original_w, original_h):
        """
        Transform coordinates from augmented image back to original image.
        
        Args:
            unit (StorageUnit): Unit to transform
            aug_idx (int): Augmentation index
            original_w (int): Original image width
            original_h (int): Original image height
        """
        # Horizontal flip (aug_idx == 1)
        if aug_idx == 1:
            unit.x1 = original_w - unit.x2
            unit.x2 = original_w - unit.x1
            
            # Also transform the mask if available
            if unit.mask is not None:
                unit.mask = cv2.flip(unit.mask.astype(np.uint8), 1).astype(bool)
        
        # 90-degree rotation (aug_idx == 2)
        elif aug_idx == 2:
            # For 90-degree rotation, swap x and y coordinates
            old_x1, old_y1, old_x2, old_y2 = unit.x1, unit.y1, unit.x2, unit.y2
            
            unit.x1 = original_h - old_y2
            unit.y1 = old_x1
            unit.x2 = original_h - old_y1
            unit.y2 = old_x2
            
            # Also transform the mask if available
            if unit.mask is not None:
                # Rotate mask back
                center = (original_w // 2, original_h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, -90, 1.0)
                unit.mask = cv2.warpAffine(
                    unit.mask.astype(np.uint8), 
                    rotation_matrix, 
                    (original_w, original_h)
                ).astype(bool)
        
        # For brightness and contrast adjustments (aug_idx == 3 or 4),
        # no coordinate transformation is needed
    
    def _non_maximum_suppression(self, units, iou_threshold=0.5):
        """
        Perform non-maximum suppression to remove duplicate detections.
        
        Args:
            units (list): List of StorageUnit objects
            iou_threshold (float): IoU threshold for suppression
            
        Returns:
            list: List of StorageUnit objects after NMS
        """
        if not units:
            return []
        
        # Sort units by confidence
        sorted_units = sorted(units, key=lambda x: x.confidence, reverse=True)
        
        # Initialize list of kept units
        kept_units = []
        
        # Perform NMS
        for unit in sorted_units:
            # Check if this unit overlaps significantly with any kept unit
            should_keep = True
            
            for kept_unit in kept_units:
                # Calculate intersection over union (IoU)
                intersection_x1 = max(unit.x1, kept_unit.x1)
                intersection_y1 = max(unit.y1, kept_unit.y1)
                intersection_x2 = min(unit.x2, kept_unit.x2)
                intersection_y2 = min(unit.y2, kept_unit.y2)
                
                if intersection_x1 < intersection_x2 and intersection_y1 < intersection_y2:
                    intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                    unit_area = unit.width * unit.height
                    kept_unit_area = kept_unit.width * kept_unit.height
                    union_area = unit_area + kept_unit_area - intersection_area
                    
                    iou = intersection_area / union_area
                    
                    if iou > iou_threshold:
                        # If using uncertainty-aware ensemble, merge the units
                        if self.use_uncertainty_aware_ensemble:
                            # Weighted average of coordinates based on confidence
                            total_confidence = unit.confidence + kept_unit.confidence
                            weight1 = unit.confidence / total_confidence
                            weight2 = kept_unit.confidence / total_confidence
                            
                            kept_unit.x1 = int(weight1 * unit.x1 + weight2 * kept_unit.x1)
                            kept_unit.y1 = int(weight1 * unit.y1 + weight2 * kept_unit.y1)
                            kept_unit.x2 = int(weight1 * unit.x2 + weight2 * kept_unit.x2)
                            kept_unit.y2 = int(weight1 * unit.y2 + weight2 * kept_unit.y2)
                            
                            # Update confidence (take the max)
                            kept_unit.confidence = max(unit.confidence, kept_unit.confidence)
                            
                            # Merge masks if available
                            if unit.mask is not None and kept_unit.mask is not None:
                                # Use logical OR to combine masks
                                kept_unit.mask = np.logical_or(unit.mask, kept_unit.mask)
                        
                        should_keep = False
                        break
            
            if should_keep:
                kept_units.append(unit)
        
        return kept_units
    
    def _detect_compartments(self, image, storage_unit, filter_small_segments=False, min_segment_width=None):
        """
        Detect compartments within a storage unit using the hybrid approach.
        
        Args:
            image (numpy.ndarray): Input image
            storage_unit (StorageUnit): Storage unit to detect compartments in
            filter_small_segments (bool): Whether to filter out small segments
            min_segment_width (int): Minimum width for segments to be included
        """
        # Crop the image to the storage unit
        x1, y1, x2, y2 = storage_unit.x1, storage_unit.y1, storage_unit.x2, storage_unit.y2
        unit_image = image[y1:y2, x1:x2]
        
        # Skip if the cropped image is too small
        if unit_image.shape[0] < 10 or unit_image.shape[1] < 10:
            return
        
        # Store original unit image dimensions
        unit_h, unit_w = unit_image.shape[:2]
        
        # If min_segment_width is not provided, calculate it based on DEFAULT_OBJECT_SIZE
        if filter_small_segments and min_segment_width is None:
            min_segment_width = int(unit_w * (DEFAULT_OBJECT_SIZE / 100))
        
        # Step 1: Use Grounding DINO for component classification
        dino_compartments = []
        
        # Get compartments from Grounding DINO
        dino_results = self.component_classifier.process_image(
            unit_image,
            detect_compartments=False,
            filter_small_segments=filter_small_segments,
            min_segment_width=min_segment_width
        )
        
        # Convert to compartments
        for dino_unit in dino_results:
            # Convert to global coordinates
            global_x1 = x1 + dino_unit.x1
            global_y1 = y1 + dino_unit.y1
            global_x2 = x1 + dino_unit.x2
            global_y2 = y1 + dino_unit.y2
            
            # Create global mask if available
            global_mask = None
            if dino_unit.mask is not None:
                global_mask = np.zeros(image.shape[:2], dtype=bool)
                global_mask[y1:y2, x1:x2][dino_unit.mask] = True
            
            # Create compartment
            compartment = StorageCompartment(
                x1=global_x1,
                y1=global_y1,
                x2=global_x2,
                y2=global_y2,
                class_name=dino_unit.class_name,
                confidence=dino_unit.confidence,
                mask=global_mask
            )
            
            dino_compartments.append(compartment)
        
        # Step 2: Use SAM for additional segmentation
        sam_compartments = []
        
        # Get all segments from SAM
        sam_segments = self._get_all_segments(unit_image)
        
        # Convert to compartments
        for sam_segment in sam_segments:
            # Convert to global coordinates
            global_x1 = x1 + sam_segment.x1
            global_y1 = y1 + sam_segment.y1
            global_x2 = x1 + sam_segment.x2
            global_y2 = y1 + sam_segment.y2
            
            # Create global mask if available
            global_mask = None
            if sam_segment.mask is not None:
                global_mask = np.zeros(image.shape[:2], dtype=bool)
                global_mask[y1:y2, x1:x2][sam_segment.mask] = True
            
            # Create compartment
            compartment = StorageCompartment(
                x1=global_x1,
                y1=global_y1,
                x2=global_x2,
                y2=global_y2,
                class_name="Compartment",  # Default class name
                confidence=sam_segment.confidence,
                mask=global_mask
            )
            
            sam_compartments.append(compartment)
        
        # Combine compartments from all detectors
        combined_compartments = []
        combined_compartments.extend(dino_compartments)
        
        # Add SAM compartments that don't overlap significantly with DINO compartments
        for sam_compartment in sam_compartments:
            # Check if this compartment overlaps significantly with any existing compartment
            should_add = True
            
            for existing_compartment in combined_compartments:
                # Calculate intersection over union (IoU)
                intersection_x1 = max(sam_compartment.x1, existing_compartment.x1)
                intersection_y1 = max(sam_compartment.y1, existing_compartment.y1)
                intersection_x2 = min(sam_compartment.x2, existing_compartment.x2)
                intersection_y2 = min(sam_compartment.y2, existing_compartment.y2)
                
                if intersection_x1 < intersection_x2 and intersection_y1 < intersection_y2:
                    intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                    sam_area = sam_compartment.width * sam_compartment.height
                    existing_area = existing_compartment.width * existing_compartment.height
                    union_area = sam_area + existing_area - intersection_area
                    
                    iou = intersection_area / union_area
                    
                    if iou > 0.5:
                        should_add = False
                        break
            
            if should_add:
                combined_compartments.append(sam_compartment)
        
        # Perform non-maximum suppression to remove duplicates
        final_compartments = self._non_maximum_suppression(combined_compartments)
        
        # Add compartments to the storage unit
        storage_unit.compartments = final_compartments
    
    def _get_all_segments(self, image):
        """
        Get all segments from an image using SAM.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            list: List of StorageUnit objects representing segments
        """
        return self.segmenter.process_image_all_segments(image)