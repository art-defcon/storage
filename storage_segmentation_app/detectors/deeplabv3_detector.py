import os
import numpy as np
import cv2
import torch
from pathlib import Path
from models import StorageUnit, StorageCompartment
from detectors.base_detector import BaseDetector
from config import DEFAULT_OBJECT_SIZE

class DeepLabV3PlusDetector(BaseDetector):
    """
    Class for detecting storage units and their compartments in images
    using DeepLabV3+ models with PyTorch.
    """
    
    def __init__(self, confidence_threshold=0.5, model_name="deeplabv3_resnet101_ade20k"):
        """
        Initialize the detector with models and parameters.
        
        Args:
            confidence_threshold (float): Minimum confidence score for detections
            model_name (str): Name of the DeepLabV3+ model to use
                Options: "deeplabv3_resnet101_ade20k", "deeplabv3_resnet50_ade20k", 
                         "deeplabv3_resnet101_voc", "deeplabv3_resnet50_voc"
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
        
        # Parse model name to get backbone and dataset
        parts = model_name.split('_')
        self.backbone = parts[1]  # resnet50 or resnet101
        self.dataset = parts[2]   # ade20k or voc
        
        # Set input size based on model requirements
        self.input_size = 512  # DeepLabV3+ typically uses 512x512 input
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load the DeepLabV3+ model for detection with segmentation capabilities."""
        try:
            import torch
            import torchvision
            from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
            
            # Check if MPS (Metal Performance Shaders) is available
            self.device = torch.device("mps" if torch.backends.mps.is_available() else 
                                      ("cuda" if torch.cuda.is_available() else "cpu"))
            
            print(f"Using device: {self.device}")
            
            # Initialize model based on backbone and dataset
            if self.backbone == "resnet101":
                if self.dataset == "ade20k":
                    # ADE20K has 150 classes
                    self.model = deeplabv3_resnet101(num_classes=150, pretrained=False)
                    weights_path = self.model_dir / "deeplabv3_resnet101_ade20k.pth"
                else:  # VOC
                    # PASCAL VOC has 21 classes (20 + background)
                    self.model = deeplabv3_resnet101(num_classes=21, pretrained=False)
                    weights_path = self.model_dir / "deeplabv3_resnet101_voc.pth"
            else:  # resnet50
                if self.dataset == "ade20k":
                    self.model = deeplabv3_resnet50(num_classes=150, pretrained=False)
                    weights_path = self.model_dir / "deeplabv3_resnet50_ade20k.pth"
                else:  # VOC
                    self.model = deeplabv3_resnet50(num_classes=21, pretrained=False)
                    weights_path = self.model_dir / "deeplabv3_resnet50_voc.pth"
            
            # Load weights if available, otherwise use pretrained weights
            if weights_path.exists():
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
                print(f"Loaded weights from {weights_path}")
            else:
                print(f"Weights file not found at {weights_path}, using pretrained weights")
                # For ADE20K
                if self.dataset == "ade20k":
                    if self.backbone == "resnet101":
                        self.model = torch.hub.load("pytorch/vision", "deeplabv3_resnet101", pretrained=True)
                        # Modify the classifier to have 150 classes for ADE20K
                        self.model.classifier[4] = torch.nn.Conv2d(256, 150, kernel_size=(1, 1), stride=(1, 1))
                    else:  # resnet50
                        self.model = torch.hub.load("pytorch/vision", "deeplabv3_resnet50", pretrained=True)
                        # Modify the classifier to have 150 classes for ADE20K
                        self.model.classifier[4] = torch.nn.Conv2d(256, 150, kernel_size=(1, 1), stride=(1, 1))
                # For PASCAL VOC
                else:  # voc
                    if self.backbone == "resnet101":
                        self.model = torch.hub.load("pytorch/vision", "deeplabv3_resnet101", pretrained=True)
                    else:  # resnet50
                        self.model = torch.hub.load("pytorch/vision", "deeplabv3_resnet50", pretrained=True)
            
            # Move model to the appropriate device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Load class mappings based on dataset
            if self.dataset == "ade20k":
                # ADE20K class mapping - focusing on storage-related classes
                # ADE20K has 150 classes, we map relevant ones to our storage classes
                self.ade20k_to_storage = {
                    # ADE20K class ID: Storage class ID
                    6: 102,    # cabinet -> Cabinet
                    23: 101,   # shelf -> Shelf
                    40: 103,   # wardrobe -> Wardrobe
                    45: 100,   # drawer -> Drawer
                    47: 104,   # box -> Storage Box
                    89: 125,   # refrigerator -> Refrigerator
                    134: 116,  # basket -> Basket
                }
            else:  # VOC
                # PASCAL VOC doesn't have many storage-specific classes
                # We'll map what's available
                self.voc_to_storage = {
                    # VOC class ID: Storage class ID
                    # Limited mapping as VOC doesn't have many storage classes
                    # We'll use generic object detection and try to classify based on shape/size
                }
            
            print(f"Loaded DeepLabV3+ model: {self.model_name}")
            
        except Exception as e:
            print(f"Error loading DeepLabV3+ model: {e}")
            self.model = None
    
    def _preprocess_image(self, image):
        """
        Preprocess the image for the DeepLabV3+ model.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Resize image to model input size
        resized_image, _, _ = self._resize_image(image)
        
        # Convert to tensor and normalize
        # DeepLabV3+ expects input normalized with ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        # Convert to float and normalize
        image_tensor = resized_image.astype(np.float32) / 255.0
        image_tensor = (image_tensor - mean) / std
        
        # Convert to PyTorch tensor and add batch dimension
        image_tensor = torch.from_numpy(image_tensor).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def process_image(self, image, detect_compartments=True, filter_small_segments=False, min_segment_width=None):
        """
        Process an image to detect storage units and their compartments using DeepLabV3+.
        
        Args:
            image (numpy.ndarray): Input image
            detect_compartments (bool): Whether to detect compartments within units
            filter_small_segments (bool): Whether to filter out small segments
            min_segment_width (int): Minimum width for segments to be included (if None, uses DEFAULT_OBJECT_SIZE% of image width)
            
        Returns:
            list: List of StorageUnit objects with detected compartments
        """
        if self.model is None:
            print("DeepLabV3+ model not loaded. Cannot process image.")
            return []
        
        # Store original image dimensions
        original_h, original_w = image.shape[:2]
        
        # If min_segment_width is not provided, calculate it based on DEFAULT_OBJECT_SIZE
        if filter_small_segments and min_segment_width is None:
            min_segment_width = int(original_w * (DEFAULT_OBJECT_SIZE / 100))
        
        # Preprocess image
        input_tensor = self._preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)["out"][0]
        
        # Get the predicted segmentation map
        output = output.argmax(0).cpu().numpy()
        
        # Resize segmentation map back to original image size
        segmentation_map = cv2.resize(output.astype(np.uint8), (original_w, original_h), 
                                     interpolation=cv2.INTER_NEAREST)
        
        storage_units = []
        
        # Process segmentation map to extract storage units
        if self.dataset == "ade20k":
            class_mapping = self.ade20k_to_storage
        else:  # voc
            class_mapping = self.voc_to_storage
        
        # Find unique class IDs in the segmentation map
        unique_classes = np.unique(segmentation_map)
        
        for class_id in unique_classes:
            # Skip background class (0)
            if class_id == 0:
                continue
            
            # Check if this class maps to a storage class
            if class_id in class_mapping:
                storage_class_id = class_mapping[class_id]
                class_name = self.storage_furniture_classes.get(storage_class_id, f"Unknown-{storage_class_id}")
                
                # Create binary mask for this class
                mask = (segmentation_map == class_id).astype(np.uint8)
                
                # Find contours in the mask
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    
                    # Skip small segments if filtering is enabled
                    if filter_small_segments and (x2 - x1) < min_segment_width:
                        print(f"Skipping small segment with width {x2 - x1} (min required: {min_segment_width})")
                        continue
                    
                    # Calculate confidence (using area ratio as a proxy for confidence)
                    contour_area = cv2.contourArea(contour)
                    bbox_area = w * h
                    confidence = min(contour_area / (bbox_area + 1e-5), 1.0)  # Avoid division by zero
                    
                    # Ensure confidence is above threshold
                    if confidence < self.confidence_threshold:
                        continue
                    
                    # Create binary mask for this instance
                    instance_mask = np.zeros((original_h, original_w), dtype=bool)
                    instance_mask[y:y+h, x:x+w] = mask[y:y+h, x:x+w] > 0
                    
                    # Print detection info for debugging
                    print(f"Detected: {class_name} (class ID: {storage_class_id}) with confidence {confidence:.2f}")
                    
                    # Create storage unit object
                    unit = StorageUnit(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        confidence=confidence,
                        class_id=storage_class_id,
                        class_name=class_name,
                        mask=instance_mask
                    )
                    
                    # Detect compartments within this unit if requested
                    if detect_compartments:
                        self._detect_compartments(image, unit, filter_small_segments, min_segment_width)
                    
                    storage_units.append(unit)
        
        return storage_units
    
    def _detect_compartments(self, image, storage_unit, filter_small_segments=False, min_segment_width=None):
        """
        Detect compartments within a storage unit using DeepLabV3+.
        
        Args:
            image (numpy.ndarray): Input image
            storage_unit (StorageUnit): Storage unit to detect compartments in
            filter_small_segments (bool): Whether to filter out small segments
            min_segment_width (int): Minimum width for segments to be included
        """
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
        
        # Preprocess the cropped image
        input_tensor = self._preprocess_image(unit_image)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)["out"][0]
        
        # Get the predicted segmentation map
        output = output.argmax(0).cpu().numpy()
        
        # Resize segmentation map back to original unit image size
        segmentation_map = cv2.resize(output.astype(np.uint8), (unit_w, unit_h), 
                                     interpolation=cv2.INTER_NEAREST)
        
        # Process segmentation map to extract compartments
        if self.dataset == "ade20k":
            class_mapping = self.ade20k_to_storage
        else:  # voc
            class_mapping = self.voc_to_storage
        
        # Find unique class IDs in the segmentation map
        unique_classes = np.unique(segmentation_map)
        
        for class_id in unique_classes:
            # Skip background class (0) and the class of the parent unit
            if class_id == 0 or (class_id in class_mapping and 
                                class_mapping[class_id] == storage_unit.class_id):
                continue
            
            # Check if this class maps to a storage class
            if class_id in class_mapping:
                storage_class_id = class_mapping[class_id]
                class_name = self.compartment_classes.get(storage_class_id, f"Unknown-{storage_class_id}")
                
                # Create binary mask for this class
                mask = (segmentation_map == class_id).astype(np.uint8)
                
                # Find contours in the mask
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # Get bounding box (in unit image coordinates)
                    cx, cy, cw, ch = cv2.boundingRect(contour)
                    cx1, cy1, cx2, cy2 = cx, cy, cx + cw, cy + ch
                    
                    # Skip small segments if filtering is enabled
                    if filter_small_segments and (cx2 - cx1) < min_segment_width:
                        print(f"Skipping small compartment with width {cx2 - cx1} (min required: {min_segment_width})")
                        continue
                    
                    # Convert to global coordinates
                    global_x1 = x1 + cx1
                    global_y1 = y1 + cy1
                    global_x2 = x1 + cx2
                    global_y2 = y1 + cy2
                    
                    # Calculate confidence (using area ratio as a proxy for confidence)
                    contour_area = cv2.contourArea(contour)
                    bbox_area = cw * ch
                    confidence = min(contour_area / (bbox_area + 1e-5), 1.0)  # Avoid division by zero
                    
                    # Ensure confidence is above threshold
                    if confidence < self.confidence_threshold:
                        continue
                    
                    # Create binary mask for this instance
                    instance_mask = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
                    instance_mask[y1:y2, x1:x2][cy:cy+ch, cx:cx+cw] = mask[cy:cy+ch, cx:cx+cw] > 0
                    
                    # Print detection info for debugging
                    print(f"Detected compartment: {class_name} (class ID: {storage_class_id}) with confidence {confidence:.2f}")
                    
                    # Create compartment object
                    compartment = StorageCompartment(
                        x1=global_x1, y1=global_y1, x2=global_x2, y2=global_y2,
                        confidence=confidence,
                        class_id=storage_class_id,
                        class_name=class_name,
                        mask=instance_mask,
                        parent_unit=storage_unit
                    )
                    
                    # Add compartment to the storage unit
                    storage_unit.add_compartment(compartment)
    
    def _get_all_segments(self, image):
        """
        Get all possible segments from DeepLabV3+ without any filtering.
        This method shows the raw segmentation output from the model.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            list: List of StorageUnit objects representing raw segments
        """
        if self.model is None:
            print("DeepLabV3+ model not loaded. Cannot process image.")
            return []
        
        # Store original image dimensions
        original_h, original_w = image.shape[:2]
        
        # Preprocess image
        input_tensor = self._preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)["out"][0]
        
        # Get the predicted segmentation map
        output = output.argmax(0).cpu().numpy()
        
        # Resize segmentation map back to original image size
        segmentation_map = cv2.resize(output.astype(np.uint8), (original_w, original_h), 
                                     interpolation=cv2.INTER_NEAREST)
        
        segments = []
        
        # Get class names based on dataset
        if self.dataset == "ade20k":
            # ADE20K has 150 classes
            num_classes = 150
            # We'll use a simplified class mapping for display
            class_names = {i: f"ADE20K-{i}" for i in range(num_classes)}
            # Add some known class names
            class_names.update({
                6: "Cabinet",
                23: "Shelf",
                40: "Wardrobe",
                45: "Drawer",
                47: "Box",
                89: "Refrigerator",
                134: "Basket",
            })
        else:  # voc
            # PASCAL VOC has 21 classes (20 + background)
            num_classes = 21
            # PASCAL VOC class names
            class_names = {
                0: "Background",
                1: "Aeroplane",
                2: "Bicycle",
                3: "Bird",
                4: "Boat",
                5: "Bottle",
                6: "Bus",
                7: "Car",
                8: "Cat",
                9: "Chair",
                10: "Cow",
                11: "Dining Table",
                12: "Dog",
                13: "Horse",
                14: "Motorbike",
                15: "Person",
                16: "Potted Plant",
                17: "Sheep",
                18: "Sofa",
                19: "Train",
                20: "TV/Monitor"
            }
        
        # Find unique class IDs in the segmentation map
        unique_classes = np.unique(segmentation_map)
        
        for class_id in unique_classes:
            # Skip background class (0)
            if class_id == 0:
                continue
            
            # Get class name
            class_name = class_names.get(class_id, f"Unknown-{class_id}")
            
            # Create binary mask for this class
            mask = (segmentation_map == class_id).astype(np.uint8)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                x1, y1, x2, y2 = x, y, x + w, y + h
                
                # Calculate confidence (using area ratio as a proxy for confidence)
                contour_area = cv2.contourArea(contour)
                bbox_area = w * h
                confidence = min(contour_area / (bbox_area + 1e-5), 1.0)  # Avoid division by zero
                
                # Create binary mask for this instance
                instance_mask = np.zeros((original_h, original_w), dtype=bool)
                instance_mask[y:y+h, x:x+w] = mask[y:y+h, x:x+w] > 0
                
                # Print detection info for debugging
                print(f"Raw DeepLabV3+ segment detected: {class_name} (class ID: {class_id}) with confidence {confidence:.2f}")
                
                # Create segment object (using StorageUnit class for consistency)
                segment = StorageUnit(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=f"DeepLabV3+ {class_name}",  # Prefix with model name
                    mask=instance_mask
                )
                
                segments.append(segment)
        
        print(f"Total raw segments detected by DeepLabV3+: {len(segments)}")
        return segments