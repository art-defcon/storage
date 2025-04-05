import os
import numpy as np
import cv2
from pathlib import Path
from models import StorageUnit, StorageCompartment
from detectors.base_detector import BaseDetector
# Remove the direct import of create_detector
from config import DEFAULT_OBJECT_SIZE

class EnsembleDetector(BaseDetector):
    """
    Class for detecting storage units and their compartments in images
    using ensemble methods:
    - Model soups (combining weights from multiple fine-tuned models)
    - Uncertainty-aware ensemble (weighting predictions by uncertainty estimates)
    - Test-time augmentation (applying multiple transformations during inference)
    """
    
    def __init__(self, confidence_threshold=0.5, ensemble_method="uncertainty", models=None):
        """
        Initialize the ensemble detector with multiple models and parameters.
        
        Args:
            confidence_threshold (float): Minimum confidence score for detections
            ensemble_method (str): Ensemble method to use ('uncertainty', 'average', 'max', 'vote')
            models (list): List of model types to include in the ensemble (if None, uses default set)
        """
        super().__init__(confidence_threshold)
        
        # Set ensemble method
        self.ensemble_method = ensemble_method
        if self.ensemble_method not in ['uncertainty', 'average', 'max', 'vote']:
            print(f"Invalid ensemble method: {ensemble_method}. Using 'uncertainty' as default.")
            self.ensemble_method = 'uncertainty'
        
        # Define default models if not provided
        if models is None:
            self.models = [
                "yolo",           # YOLOv8
                "yolo_nas_l",     # YOLO-NAS large
                "rtdetr_l",       # RT-DETR large
                "sam21",          # SAM 2.1
                "fastsam"         # FastSAM
            ]
        else:
            self.models = models
        
        # Initialize model weights for uncertainty-aware ensemble
        self.model_weights = {model: 1.0 / len(self.models) for model in self.models}
        
        # Flag to enable test-time augmentation
        self.use_test_time_augmentation = True
        
        # Load models
        self._load_models()
        
        print(f"Initialized Ensemble Detector with method: {self.ensemble_method}")
        print(f"Models: {', '.join(self.models)}")
    
    def _load_models(self):
        """Load all models for the ensemble."""
        self.detectors = {}
        
        for model_type in self.models:
            try:
                # Handle special cases for new model types
                if model_type.startswith("yolo_nas"):
                    parts = model_type.split("_")
                    size = parts[2] if len(parts) > 2 else "l"
                    from detectors.yolo_nas_detector import YOLONASDetector
                    self.detectors[model_type] = YOLONASDetector(
                        confidence_threshold=self.confidence_threshold,
                        model_size=size
                    )
                elif model_type.startswith("rtdetr"):
                    parts = model_type.split("_")
                    size = parts[1] if len(parts) > 1 else "l"
                    from detectors.rt_detr_detector import RTDETRDetector
                    self.detectors[model_type] = RTDETRDetector(
                        confidence_threshold=self.confidence_threshold,
                        model_size=size
                    )
                elif model_type == "grounding_dino":
                    from detectors.grounding_dino_detector import GroundingDINODetector
                    self.detectors[model_type] = GroundingDINODetector(
                        confidence_threshold=self.confidence_threshold
                    )
                elif model_type == "hybrid":
                    from detectors.hybrid_detector import HybridDetector
                    self.detectors[model_type] = HybridDetector(
                        confidence_threshold=self.confidence_threshold
                    )
                else:
                    # Use the factory for standard models - with lazy import
                    # Import create_detector only when needed to avoid circular imports
                    from detectors.factory import create_detector
                    self.detectors[model_type] = create_detector(
                        model_type=model_type,
                        confidence_threshold=self.confidence_threshold
                    )
                
                print(f"Loaded model: {model_type}")
            except Exception as e:
                print(f"Error loading model {model_type}: {e}")
    
    def _apply_test_time_augmentation(self, image):
        """Apply test-time augmentation to the image."""
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
        
        return augmented_images
    
    def _transform_coordinates_back(self, unit, aug_idx, original_w, original_h):
        """Transform coordinates from augmented image back to original image."""
        # Horizontal flip (aug_idx == 1)
        if aug_idx == 1:
            unit.x1, unit.x2 = original_w - unit.x2, original_w - unit.x1
            
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
    
    def _calculate_uncertainty(self, boxes):
        """Calculate uncertainty for each box based on variance in predictions."""
        if len(boxes) <= 1:
            return [0.0] * len(boxes)
        
        # Calculate mean box
        mean_box = np.mean(boxes, axis=0)
        
        # Calculate variance for each box
        variances = np.mean(np.square(boxes - mean_box), axis=1)
        
        # Normalize variances to [0, 1]
        max_var = np.max(variances) if np.max(variances) > 0 else 1.0
        uncertainties = variances / max_var
        
        return uncertainties
    
    def _non_maximum_suppression(self, units, iou_threshold=0.5):
        """Perform non-maximum suppression to remove duplicate detections."""
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
                        if self.ensemble_method == "uncertainty":
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
    
    def process_image(self, image, detect_compartments=True, filter_small_segments=False, min_segment_width=None):
        """Process an image using the ensemble of models."""
        # Store original image dimensions
        original_h, original_w = image.shape[:2]
        
        # If min_segment_width is not provided, calculate it based on DEFAULT_OBJECT_SIZE
        if filter_small_segments and min_segment_width is None:
            min_segment_width = int(original_w * (DEFAULT_OBJECT_SIZE / 100))
        
        # Apply test-time augmentation if enabled
        augmented_images = self._apply_test_time_augmentation(image) if self.use_test_time_augmentation else [image]
        
        # Initialize storage units list
        all_storage_units = []
        
        # Process each augmented image
        for aug_idx, aug_image in enumerate(augmented_images):
            print(f"Processing augmentation {aug_idx+1}/{len(augmented_images)}")
            
            # Collect predictions from all models
            model_predictions = {}
            
            for model_type, detector in self.detectors.items():
                try:
                    # Get predictions from this model
                    units = detector.process_image(
                        aug_image,
                        detect_compartments=False,  # We'll handle compartments separately
                        filter_small_segments=filter_small_segments,
                        min_segment_width=min_segment_width
                    )
                    
                    model_predictions[model_type] = units
                    print(f"Model {model_type} detected {len(units)} units")
                except Exception as e:
                    print(f"Error processing with model {model_type}: {e}")
                    model_predictions[model_type] = []
            
            # Combine predictions based on ensemble method
            combined_units = self._combine_predictions(model_predictions)
            
            # If this is not the original image (i.e., it's an augmented version),
            # transform the coordinates back to the original image space
            if aug_idx > 0:
                for unit in combined_units:
                    self._transform_coordinates_back(unit, aug_idx, original_w, original_h)
            
            # Add to the overall list
            all_storage_units.extend(combined_units)
        
        # Perform non-maximum suppression to remove duplicates
        final_units = self._non_maximum_suppression(all_storage_units)
        
        # Detect compartments if requested
        if detect_compartments:
            for unit in final_units:
                self._detect_compartments(image, unit, filter_small_segments, min_segment_width)
        
        return final_units
    
    def _combine_predictions(self, model_predictions):
        """Combine predictions from multiple models based on the ensemble method."""
        if self.ensemble_method == "uncertainty":
            return self._combine_predictions_uncertainty(model_predictions)
        elif self.ensemble_method == "average":
            return self._combine_predictions_average(model_predictions)
        elif self.ensemble_method == "max":
            return self._combine_predictions_max(model_predictions)
        else:
            # Default to uncertainty
            return self._combine_predictions_uncertainty(model_predictions)
    
    def _combine_predictions_uncertainty(self, model_predictions):
        """Combine predictions using uncertainty-aware ensemble."""
        # Collect all units from all models
        all_units = []
        for model_type, units in model_predictions.items():
            for unit in units:
                # Add model type to metadata
                if unit.metadata is None:
                    unit.metadata = {}
                unit.metadata["model_type"] = model_type
                all_units.append(unit)
        
        # Group overlapping units
        groups = []
        for unit in all_units:
            added_to_group = False
            
            for group in groups:
                # Check if this unit overlaps with any unit in the group
                for group_unit in group:
                    # Calculate IoU
                    intersection_x1 = max(unit.x1, group_unit.x1)
                    intersection_y1 = max(unit.y1, group_unit.y1)
                    intersection_x2 = min(unit.x2, group_unit.x2)
                    intersection_y2 = min(unit.y2, group_unit.y2)
                    
                    if intersection_x1 < intersection_x2 and intersection_y1 < intersection_y2:
                        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                        unit_area = unit.width * unit.height
                        group_unit_area = group_unit.width * group_unit.height
                        union_area = unit_area + group_unit_area - intersection_area
                        
                        iou = intersection_area / union_area
                        
                        if iou > 0.5:  # IoU threshold
                            group.append(unit)
                            added_to_group = True
                            break
                
                if added_to_group:
                    break
            
            if not added_to_group:
                # Create a new group
                groups.append([unit])
        
        # Combine units in each group
        combined_units = []
        
        for group in groups:
            if len(group) == 1:
                # Only one unit in the group, no need to combine
                combined_units.append(group[0])
            else:
                # Multiple units in the group, combine them
                
                # Extract bounding boxes
                boxes = np.array([[unit.x1, unit.y1, unit.x2, unit.y2] for unit in group])
                
                # Calculate uncertainty for each box
                uncertainties = self._calculate_uncertainty(boxes)
                
                # Calculate weights based on uncertainty and model weight
                weights = []
                for i, unit in enumerate(group):
                    model_type = unit.metadata.get("model_type", "unknown")
                    model_weight = self.model_weights.get(model_type, 1.0)
                    
                    # Lower uncertainty means higher weight
                    uncertainty_weight = 1.0 - uncertainties[i]
                    
                    # Combine model weight and uncertainty weight
                    weight = model_weight * uncertainty_weight * unit.confidence
                    weights.append(weight)
                
                # Normalize weights
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w / total_weight for w in weights]
                else:
                    # Equal weights if total is zero
                    weights = [1.0 / len(group)] * len(group)
                
                # Weighted average of coordinates
                x1 = int(sum(unit.x1 * weight for unit, weight in zip(group, weights)))
                y1 = int(sum(unit.y1 * weight for unit, weight in zip(group, weights)))
                x2 = int(sum(unit.x2 * weight for unit, weight in zip(group, weights)))
                y2 = int(sum(unit.y2 * weight for unit, weight in zip(group, weights)))
                
                # Weighted average of confidence
                confidence = sum(unit.confidence * weight for unit, weight in zip(group, weights))
                
                # Most common class
                class_counts = {}
                for unit in group:
                    class_name = unit.class_name
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                class_name = max(class_counts.items(), key=lambda x: x[1])[0]
                
                # Find class_id for this class_name
                class_id = None
                for unit in group:
                    if unit.class_name == class_name:
                        class_id = unit.class_id
                        break
                
                if class_id is None:
                    # Default to first unit's class_id
                    class_id = group[0].class_id
                
                # Combine masks if available
                mask = None
                masks_available = [unit.mask for unit in group if unit.mask is not None]
                if masks_available:
                    # Initialize with first mask
                    mask = masks_available[0].copy()
                    
                    # Combine with other masks
                    for m in masks_available[1:]:
                        mask = np.logical_or(mask, m)
                
                # Create combined unit
                combined_unit = StorageUnit(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                    mask=mask
                )
                
                # Add metadata about the ensemble
                combined_unit.metadata = {
                    "ensemble_method": "uncertainty",
                    "model_types": [unit.metadata.get("model_type", "unknown") for unit in group],
                    "weights": weights,
                    "uncertainties": uncertainties.tolist()
                }
                
                combined_units.append(combined_unit)
        
        return combined_units
    
    def _combine_predictions_average(self, model_predictions):
        """Combine predictions by averaging."""
        # Similar to uncertainty method but with equal weights
        all_units = []
        for model_type, units in model_predictions.items():
            for unit in units:
                if unit.metadata is None:
                    unit.metadata = {}
                unit.metadata["model_type"] = model_type
                all_units.append(unit)
        
        # Group overlapping units
        groups = []
        for unit in all_units:
            added_to_group = False
            
            for group in groups:
                for group_unit in group:
                    # Calculate IoU
                    intersection_x1 = max(unit.x1, group_unit.x1)
                    intersection_y1 = max(unit.y1, group_unit.y1)
                    intersection_x2 = min(unit.x2, group_unit.x2)
                    intersection_y2 = min(unit.y2, group_unit.y2)
                    
                    if intersection_x1 < intersection_x2 and intersection_y1 < intersection_y2:
                        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                        unit_area = unit.width * unit.height
                        group_unit_area = group_unit.width * group_unit.height
                        union_area = unit_area + group_unit_area - intersection_area
                        
                        iou = intersection_area / union_area
                        
                        if iou > 0.5:
                            group.append(unit)
                            added_to_group = True
                            break
                
                if added_to_group:
                    break
            
            if not added_to_group:
                groups.append([unit])
        
        # Combine units in each group with equal weights
        combined_units = []
        
        for group in groups:
            if len(group) == 1:
                combined_units.append(group[0])
            else:
                # Multiple units in the group, combine them with equal weights
                n = len(group)
                weights = [1.0 / n] * n
                
                # Average coordinates
                x1 = int(sum(unit.x1 for unit in group) / n)
                y1 = int(sum(unit.y1 for unit in group) / n)
                x2 = int(sum(unit.x2 for unit in group) / n)
                y2 = int(sum(unit.y2 for unit in group) / n)
                
                # Average confidence
                confidence = sum(unit.confidence for unit in group) / n
                
                # Most common class
                class_counts = {}
                for unit in group:
                    class_name = unit.class_name
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                class_name = max(class_counts.items(), key=lambda x: x[1])[0]
                
                # Find class_id for this class_name
                class_id = None
                for unit in group:
                    if unit.class_name == class_name:
                        class_id = unit.class_id
                        break
                
                if class_id is None:
                    # Default to first unit's class_id
                    class_id = group[0].class_id
                
                # Combine masks if available
                mask = None
                masks_available = [unit.mask for unit in group if unit.mask is not None]
                if masks_available:
                    # Initialize with first mask
                    mask = masks_available[0].copy()
                    
                    # Combine with other masks
                    for m in masks_available[1:]:
                        mask = np.logical_or(mask, m)
                
                # Create combined unit
                combined_unit = StorageUnit(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                    mask=mask
                )
                
                # Add metadata about the ensemble
                combined_unit.metadata = {
                    "ensemble_method": "average",
                    "model_types": [unit.metadata.get("model_type", "unknown") for unit in group],
                    "weights": weights
                }
                
                combined_units.append(combined_unit)
        
        return combined_units
    
    def _combine_predictions_max(self, model_predictions):
        """Combine predictions by taking the max confidence for each region."""
        # Similar to other methods but take the prediction with highest confidence
        all_units = []
        for model_type, units in model_predictions.items():
            for unit in units:
                if unit.metadata is None:
                    unit.metadata = {}
                unit.metadata["model_type"] = model_type
                all_units.append(unit)
        
        # Group overlapping units
        groups = []
        for unit in all_units:
            added_to_group = False
            
            for group in groups:
                for group_unit in group:
                    # Calculate IoU
                    intersection_x1 = max(unit.x1, group_unit.x1)
                    intersection_y1 = max(unit.y1, group_unit.y1)
                    intersection_x2 = min(unit.x2, group_unit.x2)
                    intersection_y2 = min(unit.y2, group_unit.y2)
                    
                    if intersection_x1 < intersection_x2 and intersection_y1 < intersection_y2:
                        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                        unit_area = unit.width * unit.height
                        group_unit_area = group_unit.width * group_unit.height
                        union_area = unit_area + group_unit_area - intersection_area
                        
                        iou = intersection_area / union_area
                        
                        if iou > 0.5:
                            group.append(unit)
                            added_to_group = True
                            break
                
                if added_to_group:
                    break
            
            if not added_to_group:
                groups.append([unit])
        
        # Take the unit with highest confidence from each group
        combined_units = []
        
        for group in groups:
            if len(group) == 1:
                combined_units.append(group[0])
            else:
                # Find the unit with highest confidence
                best_unit = max(group, key=lambda x: x.confidence)
                
                # Add metadata about the ensemble
                if best_unit.metadata is None:
                    best_unit.metadata = {}
                
                best_unit.metadata["ensemble_method"] = "max"
                best_unit.metadata["model_types"] = [unit.metadata.get("model_type", "unknown") for unit in group]
                best_unit.metadata["confidences"] = [unit.confidence for unit in group]
                
                combined_units.append(best_unit)
        
        return combined_units
    
    def _detect_compartments(self, image, storage_unit, filter_small_segments=False, min_segment_width=None):
        """Detect compartments within a storage unit."""
        # Get unit region
        x1, y1, x2, y2 = int(storage_unit.x1), int(storage_unit.y1), int(storage_unit.x2), int(storage_unit.y2)
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Skip if region is too small
        if x2 - x1 < 10 or y2 - y1 < 10:
            return
        
        # Extract unit region
        unit_image = image[y1:y2, x1:x2]
        
        # Get all segments in this region
        segments = self._get_all_segments(unit_image)
        
        # Filter small segments if requested
        if filter_small_segments and min_segment_width is not None:
            segments = [seg for seg in segments if seg.width >= min_segment_width]
        
        # Add segments as compartments to the storage unit
        for segment in segments:
            # Adjust coordinates to be relative to the original image
            segment.x1 += x1
            segment.y1 += y1
            segment.x2 += x1
            segment.y2 += y1
            
            # Add as compartment
            storage_unit.add_compartment(segment)
    
    def _get_all_segments(self, image):
        """Get all segments in an image using the ensemble of models."""
        # Use a subset of models for compartment detection
        compartment_models = ["sam21", "fastsam"]
        available_models = [m for m in compartment_models if m in self.models]
        
        if not available_models:
            # If none of the preferred models are available, use any available model
            available_models = self.models[:1]  # Just use the first model
        
        all_segments = []
        
        for model_type in available_models:
            if model_type in self.detectors:
                try:
                    # Get segments from this model
                    segments = self.detectors[model_type].process_image(
                        image,
                        detect_compartments=False
                    )
                    
                    all_segments.extend(segments)
                except Exception as e:
                    print(f"Error getting segments with model {model_type}: {e}")
        
        # Perform non-maximum suppression to remove duplicates
        return self._non_maximum_suppression(all_segments)
