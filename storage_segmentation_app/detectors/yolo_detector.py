import os
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
from models import StorageUnit, StorageCompartment
from detectors.base_detector import BaseDetector

class StorageDetector(BaseDetector):
    """
    Class for detecting storage units and their compartments in images
    using a two-stage YOLO11x-seg detection pipeline for improved accuracy.
    """
    
    def __init__(self, confidence_threshold=0.5):
        """
        Initialize the detector with models and parameters.
        
        Args:
            confidence_threshold (float): Minimum confidence score for detections
        """
        super().__init__(confidence_threshold)
        
        # Define model paths - use YOLO11x-seg model for better accuracy
        # Check both possible locations for the model file
        root_model_path = Path("../data/models/yolo11x-seg.pt")
        local_model_path = Path("data/models/yolo11x-seg.pt")
        
        # Determine which path to use
        if root_model_path.exists():
            self.unit_model_path = root_model_path
        elif local_model_path.exists():
            self.unit_model_path = local_model_path
        else:
            # Fallback to the model in the root directory
            self.unit_model_path = Path("yolo11x-seg.pt")
        
        # Use the same model for compartments
        self.compartment_model_path = self.unit_model_path
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load the YOLO11x-seg models for detection with improved segmentation capabilities."""
        try:
            # Check if the model exists at the specified path
            if self.unit_model_path.exists():
                self.unit_model = YOLO(self.unit_model_path)
                print(f"Loaded YOLO11x-seg model from {self.unit_model_path}")
            else:
                # Fallback to default YOLOv8 model if YOLO11x-seg is not found
                print(f"YOLO11x-seg model not found at {self.unit_model_path}, using default model")
                self.unit_model = YOLO("yolov8n-seg.pt")
            
            # Use the same model for compartments in this example
            # In a real application, you would train a specialized model
            self.compartment_model = self.unit_model
            
        except Exception as e:
            print(f"Error loading models: {e}")
            # Fallback to default YOLOv8 model
            self.unit_model = YOLO("yolov8n-seg.pt")
            self.compartment_model = self.unit_model
    
    def process_image(self, image, detect_compartments=True, filter_small_segments=False, min_segment_width=None):
        """
        Process an image to detect storage units and their compartments using YOLO11x-seg.
        
        Args:
            image (numpy.ndarray): Input image
            detect_compartments (bool): Whether to detect compartments within units
            filter_small_segments (bool): Whether to filter out small segments
            min_segment_width (int): Minimum width for segments to be included (if None, uses 15% of image width)
            
        Returns:
            list: List of StorageUnit objects with detected compartments
        """
        # Store original image dimensions
        original_h, original_w = image.shape[:2]
        
        # If min_segment_width is not provided, calculate it as 15% of the image width
        if filter_small_segments and min_segment_width is None:
            min_segment_width = int(original_w * 0.15)
        
        # Resize image for model input
        resized_image, scale_x, scale_y = self._resize_image(image)
        
        # Detect storage units with YOLO11x-seg
        unit_results = self.unit_model.predict(
            resized_image, 
            conf=self.confidence_threshold,
            verbose=False
        )
        
        storage_units = []
        
        # Process each detected unit
        for result in unit_results:
            boxes = result.boxes.cpu().numpy()
            masks = result.masks.cpu().numpy() if result.masks is not None else None
            
            for i, box in enumerate(boxes):
                # Get bounding box coordinates (in resized image)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Scale coordinates back to original image
                x1, y1, x2, y2 = self._scale_coordinates(x1, y1, x2, y2, scale_x, scale_y)
                
                # Skip small segments if filtering is enabled
                if filter_small_segments and (x2 - x1) < min_segment_width:
                    print(f"Skipping small segment with width {x2 - x1} (min required: {min_segment_width})")
                    continue
                
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Skip if not a storage furniture item
                if class_id not in self.storage_furniture_classes:
                    print(f"Skipping non-storage item with class ID: {class_id}")
                    continue
                
                # Get class name from class ID
                class_name = self.unit_classes.get(class_id, f"Unknown-{class_id}")
                
                # Print detection info for debugging
                print(f"Detected: {class_name} (class ID: {class_id}) with confidence {confidence:.2f}")
                
                # Get mask if available and scale it to original image size
                mask = None
                if masks is not None and i < len(masks.data):
                    # Get mask from model output
                    mask_data = masks.data[i]
                    
                    # Resize mask to original image dimensions
                    mask = cv2.resize(
                        mask_data.astype(np.uint8),
                        (original_w, original_h)
                    ).astype(bool)
                
                # Create storage unit object
                unit = StorageUnit(
                    x1=x1, y1=y1, x2=x2, y2=y2,
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
        Detect compartments within a storage unit using YOLO11x-seg.
        
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
        
        # If min_segment_width is not provided, calculate it as 15% of the unit width
        if filter_small_segments and min_segment_width is None:
            min_segment_width = int(unit_w * 0.15)
        
        # Resize unit image for model input
        resized_unit_image = cv2.resize(unit_image, (self.input_size, self.input_size))
        
        # Calculate scale factors for the unit image
        unit_scale_x = unit_w / self.input_size
        unit_scale_y = unit_h / self.input_size
        
        # Detect compartments in the resized unit image using YOLO11x-seg
        compartment_results = self.compartment_model.predict(
            resized_unit_image,
            conf=self.confidence_threshold,
            verbose=False
        )
        
        # Process each detected compartment
        for result in compartment_results:
            boxes = result.boxes.cpu().numpy()
            masks = result.masks.cpu().numpy() if result.masks is not None else None
            
            for i, box in enumerate(boxes):
                # Get bounding box coordinates (in resized unit image)
                cx1, cy1, cx2, cy2 = map(int, box.xyxy[0])
                
                # Scale coordinates back to original unit image
                cx1 = int(cx1 * unit_scale_x)
                cy1 = int(cy1 * unit_scale_y)
                cx2 = int(cx2 * unit_scale_x)
                cy2 = int(cy2 * unit_scale_y)
                
                # Skip small segments if filtering is enabled
                if filter_small_segments and (cx2 - cx1) < min_segment_width:
                    print(f"Skipping small compartment with width {cx2 - cx1} (min required: {min_segment_width})")
                    continue
                
                # Convert to global coordinates
                global_x1 = x1 + cx1
                global_y1 = y1 + cy1
                global_x2 = x1 + cx2
                global_y2 = y1 + cy2
                
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Skip if not a storage furniture item
                if class_id not in self.storage_furniture_classes:
                    print(f"Skipping non-storage compartment with class ID: {class_id}")
                    continue
                
                # Get class name from class ID
                class_name = self.compartment_classes.get(class_id, f"Unknown-{class_id}")
                
                # Print detection info for debugging
                print(f"Detected compartment: {class_name} (class ID: {class_id}) with confidence {confidence:.2f}")
                
                # Get mask if available
                mask = None
                if masks is not None and i < len(masks.data):
                    # Get mask from model output
                    mask_data = masks.data[i]
                    
                    # Resize mask to original unit dimensions
                    resized_mask = cv2.resize(
                        mask_data.astype(np.uint8),
                        (unit_w, unit_h)
                    ).astype(bool)
                    
                    # Convert to global mask
                    global_mask = np.zeros(image.shape[:2], dtype=bool)
                    global_mask[y1:y2, x1:x2] = resized_mask
                    mask = global_mask
                
                # Create compartment object
                compartment = StorageCompartment(
                    x1=global_x1, y1=global_y1, x2=global_x2, y2=global_y2,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                    mask=mask,
                    parent_unit=storage_unit
                )
                
                # Add compartment to the storage unit
                storage_unit.add_compartment(compartment)
