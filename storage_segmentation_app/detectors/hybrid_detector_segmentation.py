import numpy as np

def refine_unit_with_sam(image, unit, segmenter):
    """
    Refine a unit's segmentation using SAM.
    
    Args:
        image (numpy.ndarray): Input image
        unit: Unit to refine
        segmenter: SAM segmenter instance
        
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
    sam_results = segmenter.process_image_all_segments(cropped_image)
    
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