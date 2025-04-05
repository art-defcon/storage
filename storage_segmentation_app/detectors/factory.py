from detectors.yolo_detector import StorageDetector
from detectors.sam_detector import SAM21Detector
from detectors.fastsam_detector import FastSAMDetector
from config import DEFAULT_CONFIDENCE_THRESHOLD

def create_detector(model_type="yolo", confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """
    Factory function to create the appropriate detector based on model type.
    
    Args:
        model_type (str): Type of model to use ('yolo', 'sam21', or 'fastsam')
        confidence_threshold (float): Minimum confidence score for detections
        
    Returns:
        BaseDetector: An instance of the appropriate detector class
    """
    if model_type.lower() == "sam21":
        return SAM21Detector(confidence_threshold)
    elif model_type.lower() == "fastsam":
        return FastSAMDetector(confidence_threshold)
    else:  # Default to YOLO
        return StorageDetector(confidence_threshold)
