import numpy as np
import cv2

def has_significant_overlap(unit, unit_list, iou_threshold=0.5):
    """
    Check if a unit has significant overlap with any unit in the list.
    
    Args:
        unit: Unit to check
        unit_list: List of units to check against
        iou_threshold: IoU threshold for significant overlap
        
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

def non_maximum_suppression(units, iou_threshold=0.5, use_uncertainty_aware_ensemble=True):
    """
    Perform non-maximum suppression to remove duplicate detections.
    
    Args:
        units: List of StorageUnit objects
        iou_threshold: IoU threshold for suppression
        use_uncertainty_aware_ensemble: Whether to use uncertainty-aware ensemble
        
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
                    if use_uncertainty_aware_ensemble:
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

def transform_coordinates_back(unit, aug_idx, original_w, original_h):
    """
    Transform coordinates from augmented image back to original image.
    
    Args:
        unit: Unit to transform
        aug_idx: Augmentation index
        original_w: Original image width
        original_h: Original image height
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