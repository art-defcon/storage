import os
import numpy as np
import cv2
import torch
from pathlib import Path
from models import StorageUnit, StorageCompartment
from detectors.base_detector import BaseDetector
from config import DEFAULT_OBJECT_SIZE

class GroundingDINODetector(BaseDetector):
    """
    Class for detecting storage units and their compartments in images
    using Grounding DINO for vision-language capabilities and zero-shot detection.
    """
    
    def __init__(self, confidence_threshold=0.5, model_size="b"):
        """
        Initialize the detector with Grounding DINO model and parameters.
        
        Args:
            confidence_threshold (float): Minimum confidence score for detections
            model_size (str): Model size - 't' (tiny), 'b' (base), or 'l' (large)
        """
        super().__init__(confidence_threshold)
        
        # Set model size
        self.model_size = model_size.lower()
        if self.model_size not in ['t', 'b', 'l']:
            print(f"Invalid model size: {model_size}. Using 'b' (base) as default.")
            self.model_size = 'b'
        
        # Define model paths - check both possible locations for the model file
        model_filename = f"groundingdino_{self.model_size}.pth"
        root_model_path = Path(f"../data/models/{model_filename}")
        local_model_path = Path(f"data/models/{model_filename}")
        
        # Determine which path to use
        if root_model_path.exists():
            self.model_path = root_model_path
        elif local_model_path.exists():
            self.model_path = local_model_path
        else:
            # Use a path that will trigger the HuggingFace download
            self.model_path = None
        
        # Define text prompts for storage furniture detection
        self.storage_prompts = {
            "units": "cabinet . shelf . drawer . wardrobe . storage box . chest . sideboard . dresser . box . basket . refrigerator . chest of drawers",
            "compartments": "drawer . shelf . compartment . cabinet door . basket . tray . box . container . divider . rack"
        }
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load the Grounding DINO model for vision-language detection capabilities."""
        try:
            # Import here to avoid dependency issues if the module is not available
            from groundingdino.util.inference import load_model, load_image, predict
            
            # Load config based on model size
            if self.model_size == 't':
                config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            elif self.model_size == 'l':
                config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinL_OGC.py"
            else:  # Default to base model
                config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_OGC.py"
            
            # Load the model
            if self.model_path is not None and os.path.exists(self.model_path):
                self.model = load_model(config_path, self.model_path)
                print(f"Loaded Grounding DINO model from {self.model_path}")
            else:
                # Use HuggingFace model
                from huggingface_hub import hf_hub_download
                
                if self.model_size == 't':
                    repo_id = "ShilongLiu/GroundingDINO"
                    filename = "groundingdino_swint_ogc.pth"
                elif self.model_size == 'l':
                    repo_id = "ShilongLiu/GroundingDINO"
                    filename = "groundingdino_swinl_cogcoor.pth"
                else:
                    repo_id = "ShilongLiu/GroundingDINO"
                    filename = "groundingdino_swinb_cogcoor.pth"
                
                # Download the model
                model_path = hf_hub_download(repo_id=repo_id, filename=filename)
                self.model = load_model(config_path, model_path)
                print(f"Loaded Grounding DINO model from HuggingFace: {repo_id}/{filename}")
            
            # Store the predict function for later use
            self.predict_fn = predict
            
        except Exception as e:
            print(f"Error loading Grounding DINO model: {e}")
            print("Falling back to a placeholder implementation. Please install Grounding DINO properly.")
            
            # Create a placeholder model and predict function
            self.model = None
            self.predict_fn = self._placeholder_predict
    
    def _placeholder_predict(self, model, image, text_prompt, box_threshold, text_threshold):
        """Placeholder prediction function when Grounding DINO is not available."""
        print("WARNING: Using placeholder Grounding DINO implementation.")
        # Return empty predictions
        return [], [], []
    
    def process_image(self, image, detect_compartments=True, filter_small_segments=False, min_segment_width=None):
        """
        Process an image to detect storage units and their compartments using Grounding DINO.
        
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
        
        # If model is not available, return empty results
        if self.model is None:
            print("Grounding DINO model not available. Returning empty results.")
            return []
        
        # Convert image to RGB if it's in BGR format (OpenCV default)
        if image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Use the storage units prompt
        text_prompt = self.storage_prompts["units"]
        
        # Detect storage units with Grounding DINO
        boxes, logits, phrases = self.predict_fn(
            model=self.model,
            image=rgb_image,
            caption=text_prompt,
            box_threshold=self.confidence_threshold,
            text_threshold=self.confidence_threshold
        )
        
        storage_units = []
        
        # Process each detected unit
        for i in range(len(boxes)):
            # Get normalized box coordinates [0,1]
            x1, y1, x2, y2 = boxes[i].tolist()
            
            # Convert to absolute pixel coordinates
            x1 = int(x1 * original_w)
            y1 = int(y1 * original_h)
            x2 = int(x2 * original_w)
            y2 = int(y2 * original_h)
            
            # Skip small segments if filtering is enabled
            if filter_small_segments and (x2 - x1) < min_segment_width:
                print(f"Skipping small segment with width {x2 - x1} (min required: {min_segment_width})")
                continue
            
            confidence = float(logits[i])
            phrase = phrases[i]
            
            # Map the detected phrase to a class ID in our storage furniture classes
            class_id = None
            class_name = phrase
            
            # Try to map the phrase to one of our storage furniture classes
            for id, name in self.storage_furniture_classes.items():
                if name.lower() in phrase.lower():
                    class_id = id
                    class_name = name
                    break
            
            # If no matching class found, use a default
            if class_id is None:
                class_id = 102  # Default to Cabinet
                class_name = f"{phrase} (mapped to {self.storage_furniture_classes[class_id]})"
            
            # Print detection info for debugging
            print(f"Grounding DINO detected: {class_name} with confidence {confidence:.2f}")
            
            # Create a simple rectangular mask for the unit
            mask = np.zeros((original_h, original_w), dtype=bool)
            mask[y1:y2, x1:x2] = True
            
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
        Detect compartments within a storage unit using Grounding DINO.
        
        Args:
            image (numpy.ndarray): Input image
            storage_unit (StorageUnit): Storage unit to detect compartments in
            filter_small_segments (bool): Whether to filter out small segments
            min_segment_width (int): Minimum width for segments to be included
        """
        # If model is not available, return without adding compartments
        if self.model is None:
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
        
        # Convert image to RGB if it's in BGR format (OpenCV default)
        if unit_image.shape[2] == 3:
            rgb_unit_image = cv2.cvtColor(unit_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_unit_image = unit_image
        
        # Use the compartments prompt
        text_prompt = self.storage_prompts["compartments"]
        
        # Detect compartments with Grounding DINO
        boxes, logits, phrases = self.predict_fn(
            model=self.model,
            image=rgb_unit_image,
            caption=text_prompt,
            box_threshold=self.confidence_threshold,
            text_threshold=self.confidence_threshold
        )
        
        # Process each detected compartment
        for i in range(len(boxes)):
            # Get normalized box coordinates [0,1]
            cx1, cy1, cx2, cy2 = boxes[i].tolist()
            
            # Convert to absolute pixel coordinates within the unit
            cx1 = int(cx1 * unit_w)
            cy1 = int(cy1 * unit_h)
            cx2 = int(cx2 * unit_w)
            cy2 = int(cy2 * unit_h)
            
            # Skip small segments if filtering is enabled
            if filter_small_segments and (cx2 - cx1) < min_segment_width:
                print(f"Skipping small compartment with width {cx2 - cx1} (min required: {min_segment_width})")
                continue
            
            # Convert to global coordinates
            global_x1 = x1 + cx1
            global_y1 = y1 + cy1
            global_x2 = x1 + cx2
            global_y2 = y1 + cy2
            
            confidence = float(logits[i])
            phrase = phrases[i]
            
            # Map the detected phrase to a class ID in our compartment classes
            class_id = None
            class_name = phrase
            
            # Try to map the phrase to one of our storage furniture classes
            for id, name in self.compartment_classes.items():
                if name.lower() in phrase.lower():
                    class_id = id
                    class_name = name
                    break
            
            # If no matching class found, use a default
            if class_id is None:
                class_id = 100  # Default to Drawer
                class_name = f"{phrase} (mapped to {self.compartment_classes[class_id]})"
            
            # Print detection info for debugging
            print(f"Grounding DINO detected compartment: {class_name} with confidence {confidence:.2f}")
            
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
        Get all possible segments from Grounding DINO without any filtering.
        This method shows the raw detection output from the model.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            list: List of StorageUnit objects representing raw segments
        """
        # Store original image dimensions
        original_h, original_w = image.shape[:2]
        
        # If model is not available, return empty results
        if self.model is None:
            print("Grounding DINO model not available. Returning empty results.")
            return []
        
        # Convert image to RGB if it's in BGR format (OpenCV default)
        if image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Use a comprehensive prompt to detect all possible storage-related objects
        text_prompt = "furniture . storage . cabinet . shelf . drawer . wardrobe . box . chest . sideboard . dresser . basket . container . compartment . door . tray . divider . rack . hanger . hook . organizer . refrigerator"
        
        # Detect all segments with Grounding DINO
        boxes, logits, phrases = self.predict_fn(
            model=self.model,
            image=rgb_image,
            caption=text_prompt,
            box_threshold=0.1,  # Use a very low confidence threshold to get all segments
            text_threshold=0.1
        )
        
        segments = []
        
        # Process each detected segment
        for i in range(len(boxes)):
            # Get normalized box coordinates [0,1]
            x1, y1, x2, y2 = boxes[i].tolist()
            
            # Convert to absolute pixel coordinates
            x1 = int(x1 * original_w)
            y1 = int(y1 * original_h)
            x2 = int(x2 * original_w)
            y2 = int(y2 * original_h)
            
            confidence = float(logits[i])
            phrase = phrases[i]
            
            # Print detection info for debugging
            print(f"Raw Grounding DINO segment detected: {phrase} with confidence {confidence:.2f}")
            
            # Create a simple rectangular mask
            mask = np.zeros((original_h, original_w), dtype=bool)
            mask[y1:y2, x1:x2] = True
            
            # Create segment object (using StorageUnit class for consistency)
            segment = StorageUnit(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=confidence,
                class_id=0,  # Generic ID
                class_name=f"DINO: {phrase}",  # Prefix with model name
                mask=mask
            )
            
            segments.append(segment)
        
        print(f"Total raw segments detected by Grounding DINO: {len(segments)}")
        return segments