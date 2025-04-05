import numpy as np
import cv2
from pathlib import Path
from ultralytics import SAM
from models import StorageUnit, StorageCompartment
from detectors.base_detector import BaseDetector
from config import DEFAULT_OBJECT_SIZE

class SAM21Detector(BaseDetector):
    """
    Class for detecting storage units and their compartments in images
    using SAM 2.1 (Segment Anything Model 2.1) for improved segmentation.
    """
    
    def __init__(self, confidence_threshold=0.5):
        """
        Initialize the detector with SAM 2.1 model and parameters.
        
        Args:
            confidence_threshold (float): Minimum confidence score for detections
        """
        super().__init__(confidence_threshold)
        
        # Define model paths for SAM 2.1
        # Check both possible locations for the model file
        root_model_path = Path("../data/models/sam2.1_b.pt")
        local_model_path = Path("data/models/sam2.1_b.pt")
        
        # Determine which path to use
        if root_model_path.exists():
            self.model_path = root_model_path
        elif local_model_path.exists():
            self.model_path = local_model_path
        else:
            # Use the default SAM 2.1 model
            self.model_path = "sam2.1_b.pt"
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load the SAM 2.1 model for improved segmentation capabilities."""
        try:
            # Load SAM 2.1 model
            self.model = SAM(self.model_path)
            print(f"Loaded SAM 2.1 model from {self.model_path}")
        except Exception as e:
            print(f"Error loading SAM 2.1 model: {e}")
            # Fallback to default SAM model
            self.model = SAM("sam2.1_b.pt")
    
    def process_image(self, image, detect_compartments=True, filter_small_segments=False, min_segment_width=None):
        """
        Process an image to detect storage units and their compartments using SAM 2.1.
        
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
        
        # Resize image for model input if needed
        resized_image, scale_x, scale_y = self._resize_image(image)
        
        # Detect objects with SAM 2.1
        results = self.model.predict(
            resized_image,
            conf=self.confidence_threshold,
            verbose=False
        )
        
        storage_units = []
        
        # Process each detected object
        for result in results:
            masks = result.masks.cpu().numpy() if result.masks is not None else None
            
            if masks is None or len(masks.data) == 0:
                continue
            
            # Process each mask as a potential storage unit
            for i, mask_data in enumerate(masks.data):
                # Resize mask to original image dimensions
                mask = cv2.resize(
                    mask_data.astype(np.uint8),
                    (original_w, original_h)
                ).astype(bool)
                
                # Find bounding box from mask
                y_indices, x_indices = np.where(mask)
                if len(y_indices) == 0 or len(x_indices) == 0:
                    continue
                
                x1, y1 = np.min(x_indices), np.min(y_indices)
                x2, y2 = np.max(x_indices), np.max(y_indices)
                
                # Skip if the bounding box is too small
                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue
                
                # Skip small segments if filtering is enabled
                if filter_small_segments and (x2 - x1) < min_segment_width:
                    print(f"Skipping small segment with width {x2 - x1} (min required: {min_segment_width})")
                    continue
                
                # Assign a default class for SAM detections - use a storage furniture class
                class_id = 102  # Cabinet
                class_name = self.unit_classes.get(class_id, "Storage Unit")
                confidence = 0.9  # SAM doesn't provide confidence scores, so we use a default
                
                # Print detection info for debugging
                print(f"Detected with SAM 2.1: {class_name} with confidence {confidence:.2f}")
                
                # Create storage unit object
                unit = StorageUnit(
                    x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                    mask=mask
                )
                
                # Detect compartments within this unit if requested
                if detect_compartments:
                    self._detect_compartments(image, unit, filter_small_segments, min_segment_width)
                
                storage_units.append(unit)
        
        return storage_units
    
    def _detect_compartments(self, image, storage_unit, filter_small_segments=False, min_segment_width=None):
        """
        Detect compartments within a storage unit using SAM 2.1.
        
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
        
        # Resize unit image for model input if needed
        resized_unit_image = cv2.resize(unit_image, (self.input_size, self.input_size))
        
        # Detect compartments in the resized unit image using SAM 2.1
        compartment_results = self.model.predict(
            resized_unit_image,
            conf=self.confidence_threshold,
            verbose=False
        )
        
        # Process each detected compartment
        for result in compartment_results:
            masks = result.masks.cpu().numpy() if result.masks is not None else None
            
            if masks is None or len(masks.data) == 0:
                continue
            
            # Process each mask as a potential compartment
            for i, mask_data in enumerate(masks.data):
                # Resize mask to original unit dimensions
                resized_mask = cv2.resize(
                    mask_data.astype(np.uint8),
                    (unit_w, unit_h)
                ).astype(bool)
                
                # Find bounding box from mask
                y_indices, x_indices = np.where(resized_mask)
                if len(y_indices) == 0 or len(x_indices) == 0:
                    continue
                
                cx1, cy1 = np.min(x_indices), np.min(y_indices)
                cx2, cy2 = np.max(x_indices), np.max(y_indices)
                
                # Skip if the bounding box is too small
                if cx2 - cx1 < 5 or cy2 - cy1 < 5:
                    continue
                
                # Skip small segments if filtering is enabled
                if filter_small_segments and (cx2 - cx1) < min_segment_width:
                    print(f"Skipping small compartment with width {cx2 - cx1} (min required: {min_segment_width})")
                    continue
                
                # Convert to global coordinates
                global_x1 = x1 + cx1
                global_y1 = y1 + cy1
                global_x2 = x1 + cx2
                global_y2 = y1 + cy2
                
                # Assign a default class for SAM compartment detections - use a storage furniture class
                class_id = 118  # Box (a storage item)
                class_name = self.compartment_classes.get(class_id, "Compartment")
                confidence = 0.85  # SAM doesn't provide confidence scores, so we use a default
                
                # Create global mask
                global_mask = np.zeros(image.shape[:2], dtype=bool)
                global_mask[y1:y2, x1:x2] = resized_mask
                
                # Print detection info for debugging
                print(f"Detected compartment with SAM 2.1: {class_name} with confidence {confidence:.2f}")
                
                # Create compartment object
                compartment = StorageCompartment(
                    x1=int(global_x1), y1=int(global_y1), x2=int(global_x2), y2=int(global_y2),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                    mask=global_mask,
                    parent_unit=storage_unit
                )
                
                # Add compartment to the storage unit
                storage_unit.add_compartment(compartment)
    
    def _get_all_segments(self, image):
        """
        Get all possible segments from SAM 2.1 without any filtering.
        This method shows the raw segmentation output from the model.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            list: List of StorageUnit objects representing raw segments
        """
        # Store original image dimensions
        original_h, original_w = image.shape[:2]
        
        # Resize image for model input if needed
        resized_image, scale_x, scale_y = self._resize_image(image)
        
        # Detect all segments with SAM 2.1
        results = self.model.predict(
            resized_image,
            conf=0.1,  # Use a very low confidence threshold to get all segments
            verbose=False
        )
        
        segments = []
        segment_count = 0
        
        # Process each detected segment
        for result in results:
            masks = result.masks.cpu().numpy() if result.masks is not None else None
            
            if masks is None or len(masks.data) == 0:
                continue
            
            # Process each mask as a raw segment
            for i, mask_data in enumerate(masks.data):
                # Resize mask to original image dimensions
                mask = cv2.resize(
                    mask_data.astype(np.uint8),
                    (original_w, original_h)
                ).astype(bool)
                
                # Find bounding box from mask
                y_indices, x_indices = np.where(mask)
                if len(y_indices) == 0 or len(x_indices) == 0:
                    continue
                
                x1, y1 = np.min(x_indices), np.min(y_indices)
                x2, y2 = np.max(x_indices), np.max(y_indices)
                
                # Skip if the bounding box is too small (minimal filtering for visibility)
                if x2 - x1 < 5 or y2 - y1 < 5:
                    continue
                
                segment_count += 1
                confidence = 1.0  # Use a default confidence for visualization
                
                # Create a segment object (using StorageUnit class for consistency)
                segment = StorageUnit(
                    x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                    confidence=confidence,
                    class_id=0,  # Generic ID
                    class_name=f"SAM Segment {segment_count}",  # Generic name with counter
                    mask=mask
                )
                
                segments.append(segment)
                
                # Print detection info for debugging
                print(f"Raw SAM 2.1 segment detected: {segment.class_name}")
        
        print(f"Total raw segments detected by SAM 2.1: {segment_count}")
        return segments
