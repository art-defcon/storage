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
# Options: "YOLO", "SAM 2.1 tiny", "SAM 2.1 small", "SAM 2.1 base ⚠️", "FastSAM"
DEFAULT_SEGMENTATION_MODEL = "FastSAM"
# Get the index of the default model in the options list
MODEL_OPTIONS = ["YOLO", "SAM 2.1 tiny", "SAM 2.1 small", "SAM 2.1 base ⚠️", "FastSAM"]
DEFAULT_MODEL_INDEX = MODEL_OPTIONS.index(DEFAULT_SEGMENTATION_MODEL)

# Detection mode defaults
# Options: "Full (Units & Compartments)", "Units Only", "All Segment"
DEFAULT_DETECTION_MODE = "Full (Units & Compartments)"