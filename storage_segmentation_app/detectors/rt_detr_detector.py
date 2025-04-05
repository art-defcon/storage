import os
import numpy as np
import cv2
from pathlib import Path
from ultralytics import RTDETR
from models import StorageUnit, StorageCompartment
from detectors.base_detector import BaseDetector
from config import DEFAULT_OBJECT_SIZE

class RTDETRDetector(BaseDetector):
    """
    Class for detecting storage units and their compartments in images
    using RT-DETR (Real-Time Detection Transformer) which combines 
    transformer accuracy with YOLO speed.
    """
    
    def __init__(self, confidence_threshold=0.5, model_size="l"):
        """
        Initialize the detector with RT-DETR model and parameters.
        
        Args:
            confidence_threshold (float): Minimum confidence score for detections
            model_size (str): Model size - 's' (small), 'm' (medium), or 'l' (large)
        """
        super().__init__(confidence_threshold)
        
        # Set model size
        self.model_size = model_size.lower()
        if self.model_size not in ['s', 'm', 'l', 'x']:
            print(f"Invalid model size: {model_size}. Using 'l' (large) as default.")
            self.model_size = 'l'
        
        # Define model paths - check both possible locations for the model file
        model_filename = f"rtdetr-{self.model_size}.pt"
        root_model_path = Path(f"../data/models/{model_filename}")
        local_model_path = Path(f"data/models/{model_filename}")
        
        # Determine which path to use
        if root_model_path.exists():
            self.model_path = root_model_path
        elif local_model_path.exists():
            self.model_path = local_model_path
        else:
            # Use the default RT-DETR model from Ultralytics
            self.model_path = f"rtdetr-{self.model_size}.pt"
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load the RT-DETR model for improved detection accuracy with transformer architecture."""
        try:
            # Load RT-DETR model
            self.model = RTDETR(self.model_path)
            print(f"Loaded RT-DETR-{self.model_size.upper()} model from {self.model_path}")
        except Exception as e:
            print(f"Error loading RT-DETR model: {e}")
            # Fallback to default YOLO model
            print("Falling back to default YOLOv8 model")
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")
    
    def process_image(self, image, detect_compartments=True, filter_small_segments=False, min_segment_width=None):
        """
        Process an image to detect storage units and their compartments using RT-DETR.
        
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
        
        # Resize image for model input
        resized_image, scale_x, scale_y = self._resize_image(image)
        
        # Detect storage units with RT-DETR
        unit_results = self.model.predict(
            resized_image, 
            conf=self.confidence_threshold,
            verbose=False
        )
        
        storage_units = []
        
        # Process each detected unit
        for result in unit_results:
            boxes = result.boxes.cpu().numpy()
            
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
                print(f"RT-DETR detected: {class_name} (class ID: {class_id}) with confidence {confidence:.2f}")
                
                # Create storage unit object - RT-DETR doesn't provide masks by default
                # so we'll create a simple rectangular mask
                mask = np.zeros((original_h, original_w), dtype=bool)
                mask[y1:y2, x1:x2] = True
                
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
        Detect compartments within a storage unit using RT-DETR.
        
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
        
        # Resize unit image for model input
        resized_unit_image = cv2.resize(unit_image, (self.input_size, self.input_size))
        
        # Calculate scale factors for the unit image
        unit_scale_x = unit_w / self.input_size
        unit_scale_y = unit_h / self.input_size
        
        # Detect compartments in the resized unit image using RT-DETR
        compartment_results = self.model.predict(
            resized_unit_image,
            conf=self.confidence_threshold,
            verbose=False
        )
        
        # Process each detected compartment
        for result in compartment_results:
            boxes = result.boxes.cpu().numpy()
            
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
                print(f"RT-DETR detected compartment: {class_name} (class ID: {class_id}) with confidence {confidence:.2f}")
                
                # Create a simple rectangular mask for the compartment
                global_mask = np.zeros(image.shape[:2], dtype=bool)
                global_mask[global_y1:global_y2, global_x1:global_x2] = True
                
                # Create compartment object
                compartment = StorageCompartment(
                    x1=global_x1, y1=global_y1, x2=global_x2, y2=global_y2,
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
        Get all possible segments from RT-DETR without any filtering.
        This method shows the raw detection output from the model.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            list: List of StorageUnit objects representing raw segments
        """
        # Store original image dimensions
        original_h, original_w = image.shape[:2]
        
        # Resize image for model input
        resized_image, scale_x, scale_y = self._resize_image(image)
        
        # Detect all segments with RT-DETR
        results = self.model.predict(
            resized_image,
            conf=0.1,  # Use a very low confidence threshold to get all segments
            verbose=False
        )
        
        segments = []
        
        # Process each detected segment
        for result in results:
            boxes = result.boxes.cpu().numpy()
            classes = result.names
            
            for i, box in enumerate(boxes):
                # Get bounding box coordinates (in resized image)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Scale coordinates back to original image
                x1, y1, x2, y2 = self._scale_coordinates(x1, y1, x2, y2, scale_x, scale_y)
                
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Get the actual class name from model
                class_name = classes.get(class_id, f"Unknown-{class_id}")
                
                # Print detection info for debugging
                print(f"Raw RT-DETR segment detected: {class_name} (class ID: {class_id}) with confidence {confidence:.2f}")
                
                # Create a simple rectangular mask
                mask = np.zeros((original_h, original_w), dtype=bool)
                mask[y1:y2, x1:x2] = True
                
                # Create segment object (using StorageUnit class for consistency)
                segment = StorageUnit(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=f"RT-DETR {class_name}",  # Prefix with model name
                    mask=mask
                )
                
                segments.append(segment)
        
        print(f"Total raw segments detected by RT-DETR: {len(segments)}")
        return segments