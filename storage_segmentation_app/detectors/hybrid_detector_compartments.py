import numpy as np
from models import StorageCompartment
from detectors.hybrid_detector_utils import has_significant_overlap, non_maximum_suppression

def detect_compartments(image, storage_unit, component_classifier, segmenter, 
                        filter_small_segments=False, min_segment_width=None, 
                        default_object_size=None, use_uncertainty_aware_ensemble=True):
    """
    Detect compartments within a storage unit using the hybrid approach.
    
    Args:
        image (numpy.ndarray): Input image
        storage_unit: Storage unit to detect compartments in
        component_classifier: Component classifier instance
        segmenter: Segmenter instance
        filter_small_segments (bool): Whether to filter out small segments
        min_segment_width (int): Minimum width for segments to be included
        default_object_size: Default object size percentage
        use_uncertainty_aware_ensemble: Whether to use uncertainty-aware ensemble
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
    if filter_small_segments and min_segment_width is None and default_object_size is not None:
        min_segment_width = int(unit_w * (default_object_size / 100))
    
    # Step 1: Use Grounding DINO for component classification
    dino_compartments = []
    
    # Get compartments from Grounding DINO
    dino_results = component_classifier.process_image(
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
    sam_segments = segmenter.process_image_all_segments(unit_image)
    
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
    final_compartments = non_maximum_suppression(
        combined_compartments, 
        iou_threshold=0.5, 
        use_uncertainty_aware_ensemble=use_uncertainty_aware_ensemble
    )
    
    # Add compartments to the storage unit
    storage_unit.compartments = final_compartments