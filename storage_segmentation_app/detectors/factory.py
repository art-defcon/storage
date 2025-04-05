from detectors.yolo_detector import StorageDetector
from detectors.sam_detector import SAM21Detector
from detectors.fastsam_detector import FastSAMDetector
from detectors.detectron2_detector import Detectron2Detector
from detectors.deeplabv3_detector import DeepLabV3PlusDetector
from config import DEFAULT_CONFIDENCE_THRESHOLD

def create_detector(model_type="yolo", confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """
    Factory function to create the appropriate detector based on model type.
    
    Args:
        model_type (str): Type of model to use ('yolo', 'sam21', 'fastsam', 'detectron2_fpn', 'detectron2_c4',
                          'deeplabv3_resnet101_ade20k', 'deeplabv3_resnet50_ade20k', 
                          'deeplabv3_resnet101_voc', 'deeplabv3_resnet50_voc')
        confidence_threshold (float): Minimum confidence score for detections
        
    Returns:
        BaseDetector: An instance of the appropriate detector class
    """
    if model_type.lower() == "sam21":
        return SAM21Detector(confidence_threshold)
    elif model_type.lower() == "fastsam":
        return FastSAMDetector(confidence_threshold)
    elif model_type.lower() == "detectron2_fpn":
        return Detectron2Detector(confidence_threshold, model_name="mask_rcnn_R_50_FPN_1x")
    elif model_type.lower() == "detectron2_c4":
        return Detectron2Detector(confidence_threshold, model_name="mask_rcnn_R_50_C4_1x")
    elif model_type.lower() == "deeplabv3_resnet101_ade20k":
        return DeepLabV3PlusDetector(confidence_threshold, model_name="deeplabv3_resnet101_ade20k")
    elif model_type.lower() == "deeplabv3_resnet50_ade20k":
        return DeepLabV3PlusDetector(confidence_threshold, model_name="deeplabv3_resnet50_ade20k")
    elif model_type.lower() == "deeplabv3_resnet101_voc":
        return DeepLabV3PlusDetector(confidence_threshold, model_name="deeplabv3_resnet101_voc")
    elif model_type.lower() == "deeplabv3_resnet50_voc":
        return DeepLabV3PlusDetector(confidence_threshold, model_name="deeplabv3_resnet50_voc")
    else:  # Default to YOLO
        return StorageDetector(confidence_threshold)
