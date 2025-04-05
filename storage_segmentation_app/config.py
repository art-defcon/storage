"""
Configuration file for the Storage Segmentation App.
Contains default values for UI components and other settings.
"""

# Default UI settings
DEFAULT_CONFIDENCE_THRESHOLD = 0.8
DEFAULT_OBJECT_SIZE = 20
# Note: The max value for contour smoothing in the UI is 0.1, so we're using 0.1 instead of 1.0
DEFAULT_CONTOUR_SMOOTHING_FACTOR = 0.1
DEFAULT_CORNER_ROUNDING_ITERATIONS = 0

# Model selection defaults
# Options: "YOLO", "SAM 2.1 tiny", "SAM 2.1 small", "SAM 2.1 base ⚠️", "FastSAM", 
#          "YOLO-NAS S", "YOLO-NAS M", "YOLO-NAS L",
#          "RT-DETR S", "RT-DETR M", "RT-DETR L", "RT-DETR X",
#          "Grounding DINO", 
#          "Hybrid Pipeline",
#          "Ensemble (Uncertainty)", "Ensemble (Average)", "Ensemble (Max)",
#          "Mask R-CNN (FPN)", "Mask R-CNN (C4)",
#          "DeepLabV3+ ResNet101 (ADE20K)", "DeepLabV3+ ResNet50 (ADE20K)",
#          "DeepLabV3+ ResNet101 (VOC)", "DeepLabV3+ ResNet50 (VOC)"
DEFAULT_SEGMENTATION_MODEL = "Hybrid Pipeline"

# Get the index of the default model in the options list
MODEL_OPTIONS = [
    # Original models
    "YOLO", 
    "SAM 2.1 tiny", 
    "SAM 2.1 small", 
    "SAM 2.1 base ⚠️", 
    "FastSAM", 
    
    # New models
    "YOLO-NAS S",
    "YOLO-NAS M",
    "YOLO-NAS L",
    "RT-DETR S",
    "RT-DETR M",
    "RT-DETR L",
    "RT-DETR X",
    "Grounding DINO",
    
    # Advanced pipelines
    "Hybrid Pipeline",
    "Ensemble (Uncertainty)",
    "Ensemble (Average)",
    "Ensemble (Max)",
    
    # Original models (continued)
    "Mask R-CNN (FPN)", 
    "Mask R-CNN (C4)",
    "DeepLabV3+ ResNet101 (ADE20K)",
    "DeepLabV3+ ResNet50 (ADE20K)",
    "DeepLabV3+ ResNet101 (VOC)",
    "DeepLabV3+ ResNet50 (VOC)"
]

DEFAULT_MODEL_INDEX = MODEL_OPTIONS.index(DEFAULT_SEGMENTATION_MODEL)

# Detection mode defaults
# Options: "Full (Units & Compartments)", "Units Only", "All Segment"
DEFAULT_DETECTION_MODE = "Full (Units & Compartments)"

# Dynamic confidence thresholds based on furniture type
DYNAMIC_CONFIDENCE_THRESHOLDS = {
    "Drawer": 0.65,      # Drawers need higher confidence due to similar appearance
    "Shelf": 0.55,       # Shelves are usually easier to detect
    "Cabinet": 0.60,     # Cabinets can be complex
    "Wardrobe": 0.70,    # Wardrobes are large but can be confused with other furniture
    "Storage Box": 0.65, # Boxes can be confused with other objects
    "Chest": 0.65,       # Chests can be confused with other furniture
    "Sideboard": 0.70,   # Sideboards can be complex
    "Dresser": 0.70,     # Dressers can be complex
    "Box": 0.65,         # Boxes can be confused with other objects
    "Basket": 0.65,      # Baskets can be confused with other objects
    "Refrigerator": 0.75, # Refrigerators are distinctive
    "Chest of Drawers": 0.70  # Complex furniture
}

# Ensemble settings
DEFAULT_ENSEMBLE_METHOD = "uncertainty"  # Options: "uncertainty", "average", "max", "vote"
DEFAULT_ENSEMBLE_MODELS = [
    "yolo",           # YOLOv8
    "yolo_nas_l",     # YOLO-NAS large
    "rtdetr_l",       # RT-DETR large
    "sam21",          # SAM 2.1
    "fastsam"         # FastSAM
]

# Test-time augmentation settings
USE_TEST_TIME_AUGMENTATION = True