import os
import numpy as np
import cv2
from pathlib import Path
import torch
from models import StorageUnit, StorageCompartment
from detectors.base_detector import BaseDetector
from config import DEFAULT_OBJECT_SIZE

class Detectron2Detector(BaseDetector):
    """
    Class for detecting storage units and their compartments in images
    using Detectron2's Mask R-CNN models.
    """
    
    def __init__(self, confidence_threshold=0.5, model_name="mask_rcnn_R_50_FPN_1x"):
        """
        Initialize the detector with models and parameters.
        
        Args:
            confidence_threshold (float): Minimum confidence score for detections
            model_name (str): Name of the Detectron2 model to use
                Options: "mask_rcnn_R_50_FPN_1x", "mask_rcnn_R_50_C4_1x"
        """
        super().__init__(confidence_threshold)
        
        # Store the model name
        self.model_name = model_name
        
        # Define model paths - check both possible locations for the model file
        root_model_dir = Path("../data/models")
        local_model_dir = Path("data/models")
        
        # Determine which path to use
        if root_model_dir.exists():
            self.model_dir = root_model_dir
        else:
            self.model_dir = local_model_dir
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load the Detectron2 Mask R-CNN models for detection with segmentation capabilities."""
        try:
            # Import detectron2 here to avoid import errors if the library is not installed
            from detectron2 import model_zoo
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg
            
            # Create config
            self.cfg = get_cfg()
            
            # Set the model configuration based on the selected model
            if self.model_name == "mask_rcnn_R_50_C4_1x":
                config_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml"
            else:  # Default to mask_rcnn_R_50_FPN_1x
                config_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
            
            # Load the configuration from model_zoo
            self.cfg.merge_from_file(model_zoo.get_config_file(config_path))
            
            # Set the model weights
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)
            
            # Set the confidence threshold
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
            
            # Create the predictor
            self.predictor = DefaultPredictor(self.cfg)
            
            print(f"Loaded Detectron2 model: {self.model_name}")
            
        except Exception as e:
            print(f"Error loading Detectron2 model: {e}")
            self.predictor = None
    
    def process_image(self, image, detect_compartments=True, filter_small_segments=False, min_segment_width=None):
        """
        Process an image to detect storage units and their compartments using Detectron2.
        
        Args:
            image (numpy.ndarray): Input image
            detect_compartments (bool): Whether to detect compartments within units
            filter_small_segments (bool): Whether to filter out small segments
            min_segment_width (int): Minimum width for segments to be included (if None, uses DEFAULT_OBJECT_SIZE% of image width)
            
        Returns:
            list: List of StorageUnit objects with detected compartments
        """
        if self.predictor is None:
            print("Detectron2 model not loaded. Cannot process image.")
            return []
        
        # Store original image dimensions
        original_h, original_w = image.shape[:2]
        
        # If min_segment_width is not provided, calculate it based on DEFAULT_OBJECT_SIZE
        if filter_small_segments and min_segment_width is None:
            min_segment_width = int(original_w * (DEFAULT_OBJECT_SIZE / 100))
        
        # Run inference with Detectron2
        outputs = self.predictor(image)
        
        # Extract predictions
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        masks = instances.pred_masks.numpy() if instances.has("pred_masks") else None
        
        storage_units = []
        
        # Process each detected instance
        for i in range(len(boxes)):
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, boxes[i])
            
            # Skip small segments if filtering is enabled
            if filter_small_segments and (x2 - x1) < min_segment_width:
                print(f"Skipping small segment with width {x2 - x1} (min required: {min_segment_width})")
                continue
            
            confidence = float(scores[i])
            class_id = int(classes[i])
            
            # Map COCO class IDs to our storage furniture classes
            # Detectron2 uses COCO dataset class IDs
            if class_id in self.storage_furniture_classes:
                class_name = self.storage_furniture_classes[class_id]
            else:
                # Skip if not a storage furniture item
                print(f"Skipping non-storage item with class ID: {class_id}")
                continue
            
            # Print detection info for debugging
            print(f"Detected: {class_name} (class ID: {class_id}) with confidence {confidence:.2f}")
            
            # Get mask if available
            mask = None
            if masks is not None:
                mask = masks[i].astype(bool)
            
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
        Detect compartments within a storage unit using Detectron2.
        
        Args:
            image (numpy.ndarray): Input image
            storage_unit (StorageUnit): Storage unit to detect compartments in
            filter_small_segments (bool): Whether to filter out small segments
            min_segment_width (int): Minimum width for segments to be included
        """
        if self.predictor is None:
            return
        
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
        
        # Run inference with Detectron2 on the cropped image
        outputs = self.predictor(unit_image)
        
        # Extract predictions
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        masks = instances.pred_masks.numpy() if instances.has("pred_masks") else None
        
        # Process each detected compartment
        for i in range(len(boxes)):
            # Get bounding box coordinates (in unit image)
            cx1, cy1, cx2, cy2 = map(int, boxes[i])
            
            # Skip small segments if filtering is enabled
            if filter_small_segments and (cx2 - cx1) < min_segment_width:
                print(f"Skipping small compartment with width {cx2 - cx1} (min required: {min_segment_width})")
                continue
            
            # Convert to global coordinates
            global_x1 = x1 + cx1
            global_y1 = y1 + cy1
            global_x2 = x1 + cx2
            global_y2 = y1 + cy2
            
            confidence = float(scores[i])
            class_id = int(classes[i])
            
            # Map COCO class IDs to our storage furniture classes
            if class_id in self.storage_furniture_classes:
                class_name = self.compartment_classes[class_id]
            else:
                # Skip if not a storage furniture item
                print(f"Skipping non-storage compartment with class ID: {class_id}")
                continue
            
            # Print detection info for debugging
            print(f"Detected compartment: {class_name} (class ID: {class_id}) with confidence {confidence:.2f}")
            
            # Get mask if available
            mask = None
            if masks is not None:
                # Get mask from model output
                mask_data = masks[i]
                
                # Create global mask
                global_mask = np.zeros(image.shape[:2], dtype=bool)
                
                # Resize mask to original unit dimensions if needed
                if mask_data.shape != (unit_h, unit_w):
                    resized_mask = cv2.resize(
                        mask_data.astype(np.uint8),
                        (unit_w, unit_h)
                    ).astype(bool)
                    global_mask[y1:y2, x1:x2] = resized_mask
                else:
                    global_mask[y1:y2, x1:x2] = mask_data
                
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
    
    def _get_all_segments(self, image):
        """
        Get all possible segments from Detectron2 without any filtering.
        This method shows the raw segmentation output from the model.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            list: List of StorageUnit objects representing raw segments
        """
        if self.predictor is None:
            print("Detectron2 model not loaded. Cannot process image.")
            return []
        
        # Run inference with Detectron2 using a lower confidence threshold
        # to get more segments
        original_threshold = self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
        self.predictor = DefaultPredictor(self.cfg)
        
        outputs = self.predictor(image)
        
        # Reset the confidence threshold
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = original_threshold
        self.predictor = DefaultPredictor(self.cfg)
        
        # Extract predictions
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        masks = instances.pred_masks.numpy() if instances.has("pred_masks") else None
        
        segments = []
        
        # Get COCO class names
        from detectron2.data import MetadataCatalog
        metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        class_names = metadata.get("thing_classes", None)
        
        # Process each detected segment
        for i in range(len(boxes)):
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, boxes[i])
            
            confidence = float(scores[i])
            class_id = int(classes[i])
            
            # Get the actual class name from Detectron2 model
            if class_names and class_id < len(class_names):
                class_name = class_names[class_id]
            else:
                class_name = f"Unknown-{class_id}"
            
            # Print detection info for debugging
            print(f"Raw Detectron2 segment detected: {class_name} (class ID: {class_id}) with confidence {confidence:.2f}")
            
            # Get mask if available
            mask = None
            if masks is not None:
                mask = masks[i].astype(bool)
            
            # Create segment object (using StorageUnit class for consistency)
            segment = StorageUnit(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=confidence,
                class_id=class_id,
                class_name=f"Detectron2 {class_name}",  # Prefix with model name
                mask=mask
            )
            
            segments.append(segment)
        
        print(f"Total raw segments detected by Detectron2: {len(segments)}")
        return segments