import os
import numpy as np
import cv2
from ultralytics import YOLO, SAM, FastSAM
from pathlib import Path
from abc import ABC, abstractmethod

from models import StorageUnit, StorageCompartment

class BaseDetector(ABC):
    """
    Abstract base class for storage unit and compartment detection.
    """
    
    def __init__(self, confidence_threshold=0.5):
        """
        Initialize the detector with parameters.
        
        Args:
            confidence_threshold (float): Minimum confidence score for detections
        """
        self.confidence_threshold = confidence_threshold
        self.input_size = 640
        
        # Define storage furniture classes - ONLY include storage furniture items
        self.storage_furniture_classes = {
            # COCO classes that map to storage furniture
            75: "Vase",        # vase (COCO class)
            73: "Bookshelf",   # book (COCO class, will be treated as bookshelf)
            
            # Storage-specific classes
            100: "Drawer",
            101: "Shelf",
            102: "Cabinet",
            103: "Wardrobe",
            104: "Storage Box",
            105: "Chest",
            106: "Sideboard",
            109: "Dresser",
            118: "Box",
            116: "Basket",
            125: "Refrigerator",
            126: "Chest of Drawers"
        }
        
        # Use the storage furniture classes for both unit and compartment detection
        self.unit_classes = self.storage_furniture_classes.copy()
        self.compartment_classes = self.storage_furniture_classes.copy()
        
        # Keep the full list of storage-specific classes for reference
        # but we'll only use the ones in storage_furniture_classes for detection
        self.storage_specific_classes = {
            100: "Drawer",
            101: "Shelf",
            102: "Cabinet",
            103: "Wardrobe",
            104: "Storage Box",
            105: "Chest",
            106: "Sideboard",
            107: "Console Table",
            108: "Nightstand",
            109: "Dresser",
            110: "Hutch",
            111: "Credenza",
            112: "Cubby",
            113: "Bin",
            114: "Compartment",
            115: "Cabinet Door",
            116: "Basket",
            117: "Tray",
            118: "Box",
            119: "Container",
            120: "Divider",
            121: "Rack",
            122: "Hanger",
            123: "Hook",
            124: "Organizer",
            125: "Refrigerator",
            126: "Chest of Drawers"
        }
    
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
    def process_image(self, image, detect_compartments=True):
        """
        Process an image to detect storage units and their compartments.
        
        Args:
            image (numpy.ndarray): Input image
            detect_compartments (bool): Whether to detect compartments within units
            
        Returns:
            list: List of StorageUnit objects with detected compartments
        """
        pass
    
    @abstractmethod
    def _detect_compartments(self, image, storage_unit):
        """
        Detect compartments within a storage unit.
        
        Args:
            image (numpy.ndarray): Input image
            storage_unit (StorageUnit): Storage unit to detect compartments in
        """
        pass


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
    
    def process_image(self, image, detect_compartments=True):
        """
        Process an image to detect storage units and their compartments using YOLO11x-seg.
        
        Args:
            image (numpy.ndarray): Input image
            detect_compartments (bool): Whether to detect compartments within units
            
        Returns:
            list: List of StorageUnit objects with detected compartments
        """
        # Store original image dimensions
        original_h, original_w = image.shape[:2]
        
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
                    self._detect_compartments(image, unit)
                
                storage_units.append(unit)
        
        return storage_units
    
    def _detect_compartments(self, image, storage_unit):
        """
        Detect compartments within a storage unit using YOLO11x-seg.
        
        Args:
            image (numpy.ndarray): Input image
            storage_unit (StorageUnit): Storage unit to detect compartments in
        """
        # Crop the image to the storage unit
        x1, y1, x2, y2 = storage_unit.x1, storage_unit.y1, storage_unit.x2, storage_unit.y2
        unit_image = image[y1:y2, x1:x2]
        
        # Skip if the cropped image is too small
        if unit_image.shape[0] < 10 or unit_image.shape[1] < 10:
            return
        
        # Store original unit image dimensions
        unit_h, unit_w = unit_image.shape[:2]
        
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


def create_detector(model_type="yolo", confidence_threshold=0.5):
    """
    Factory function to create the appropriate detector based on model type.
    
    Args:
        model_type (str): Type of model to use ('yolo', 'sam21', or 'fastsam')
        confidence_threshold (float): Minimum confidence score for detections
        
    Returns:
        BaseDetector: An instance of the appropriate detector class
    """
    if model_type.lower() == "sam21":
        from ultralytics import SAM
        return SAM21Detector(confidence_threshold)
    elif model_type.lower() == "fastsam":
        from ultralytics import FastSAM
        return FastSAMDetector(confidence_threshold)
    else:  # Default to YOLO
        return StorageDetector(confidence_threshold)

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
    
    def process_image(self, image, detect_compartments=True):
        """
        Process an image to detect storage units and their compartments using SAM 2.1.
        
        Args:
            image (numpy.ndarray): Input image
            detect_compartments (bool): Whether to detect compartments within units
            
        Returns:
            list: List of StorageUnit objects with detected compartments
        """
        # Store original image dimensions
        original_h, original_w = image.shape[:2]
        
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
                    self._detect_compartments(image, unit)
                
                storage_units.append(unit)
        
        return storage_units
    
    def _detect_compartments(self, image, storage_unit):
        """
        Detect compartments within a storage unit using SAM 2.1.
        
        Args:
            image (numpy.ndarray): Input image
            storage_unit (StorageUnit): Storage unit to detect compartments in
        """
        # Crop the image to the storage unit
        x1, y1, x2, y2 = storage_unit.x1, storage_unit.y1, storage_unit.x2, storage_unit.y2
        unit_image = image[y1:y2, x1:x2]
        
        # Skip if the cropped image is too small
        if unit_image.shape[0] < 10 or unit_image.shape[1] < 10:
            return
        
        # Store original unit image dimensions
        unit_h, unit_w = unit_image.shape[:2]
        
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


class FastSAMDetector(BaseDetector):
    """
    Class for detecting storage units and their compartments in images
    using FastSAM for efficient segmentation.
    """
    
    def __init__(self, confidence_threshold=0.5):
        """
        Initialize the detector with FastSAM model and parameters.
        
        Args:
            confidence_threshold (float): Minimum confidence score for detections
        """
        super().__init__(confidence_threshold)
        
        # Define model paths for FastSAM
        # Check both possible locations for the model file
        root_model_path = Path("../data/models/FastSAM-s.pt")
        local_model_path = Path("data/models/FastSAM-s.pt")
        
        # Determine which path to use
        if root_model_path.exists():
            self.model_path = root_model_path
        elif local_model_path.exists():
            self.model_path = local_model_path
        else:
            # Use the default FastSAM model
            self.model_path = "FastSAM-s.pt"
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load the FastSAM model for efficient segmentation capabilities."""
        try:
            # Load FastSAM model
            self.model = FastSAM(self.model_path)
            print(f"Loaded FastSAM model from {self.model_path}")
        except Exception as e:
            print(f"Error loading FastSAM model: {e}")
            # Fallback to default FastSAM model
            self.model = FastSAM("FastSAM-s.pt")
    
    def process_image(self, image, detect_compartments=True):
        """
        Process an image to detect storage units and their compartments using FastSAM.
        
        Args:
            image (numpy.ndarray): Input image
            detect_compartments (bool): Whether to detect compartments within units
            
        Returns:
            list: List of StorageUnit objects with detected compartments
        """
        # Store original image dimensions
        original_h, original_w = image.shape[:2]
        
        # Resize image for model input if needed
        resized_image, scale_x, scale_y = self._resize_image(image)
        
        # Detect objects with FastSAM
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
                
                # Assign a default class for FastSAM detections - use a storage furniture class
                class_id = 102  # Cabinet
                class_name = self.unit_classes.get(class_id, "Storage Unit")
                confidence = 0.9  # FastSAM doesn't provide confidence scores, so we use a default
                
                # Print detection info for debugging
                print(f"Detected with FastSAM: {class_name} with confidence {confidence:.2f}")
                
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
                    self._detect_compartments(image, unit)
                
                storage_units.append(unit)
        
        return storage_units
    
    def _detect_compartments(self, image, storage_unit):
        """
        Detect compartments within a storage unit using FastSAM.
        
        Args:
            image (numpy.ndarray): Input image
            storage_unit (StorageUnit): Storage unit to detect compartments in
        """
        # Crop the image to the storage unit
        x1, y1, x2, y2 = storage_unit.x1, storage_unit.y1, storage_unit.x2, storage_unit.y2
        unit_image = image[y1:y2, x1:x2]
        
        # Skip if the cropped image is too small
        if unit_image.shape[0] < 10 or unit_image.shape[1] < 10:
            return
        
        # Store original unit image dimensions
        unit_h, unit_w = unit_image.shape[:2]
        
        # Resize unit image for model input if needed
        resized_unit_image = cv2.resize(unit_image, (self.input_size, self.input_size))
        
        # Detect compartments in the resized unit image using FastSAM
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
