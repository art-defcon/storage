from detectors.yolo_detector import StorageDetector
from detectors.sam_detector import SAM21Detector
from detectors.fastsam_detector import FastSAMDetector
from detectors.detectron2_detector import Detectron2Detector
from detectors.deeplabv3_detector import DeepLabV3PlusDetector
from detectors.yolo_nas_detector import YOLONASDetector
from detectors.rt_detr_detector import RTDETRDetector
from detectors.grounding_dino_detector import GroundingDINODetector
from detectors.hybrid_detector import HybridDetector
from detectors.ensemble_detector import EnsembleDetector
from config import DEFAULT_CONFIDENCE_THRESHOLD

def create_detector(model_type="yolo", confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, **kwargs):
    """
    Factory function to create the appropriate detector based on model type.
    
    Args:
        model_type (str): Type of model to use. Options include:
            - 'yolo': YOLOv8 detector
            - 'sam21': SAM 2.1 detector
            - 'fastsam': FastSAM detector
            - 'yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l': YOLO-NAS detectors (small, medium, large)
            - 'rtdetr_s', 'rtdetr_m', 'rtdetr_l', 'rtdetr_x': RT-DETR detectors (small, medium, large, xlarge)
            - 'grounding_dino_t', 'grounding_dino_b', 'grounding_dino_l': Grounding DINO detectors (tiny, base, large)
            - 'hybrid': Hybrid detector combining YOLO-NAS, Grounding DINO, and SAM 2.1
            - 'ensemble': Ensemble detector combining multiple models
            - 'detectron2_fpn', 'detectron2_c4': Detectron2 detectors
            - 'deeplabv3_resnet101_ade20k', 'deeplabv3_resnet50_ade20k', 
              'deeplabv3_resnet101_voc', 'deeplabv3_resnet50_voc': DeepLabV3+ detectors
        confidence_threshold (float): Minimum confidence score for detections
        **kwargs: Additional keyword arguments for specific detector types
        
    Returns:
        BaseDetector: An instance of the appropriate detector class
    """
    model_type = model_type.lower()
    
    # SAM 2.1 detector
    if model_type == "sam21":
        return SAM21Detector(confidence_threshold)
    
    # FastSAM detector
    elif model_type == "fastsam":
        return FastSAMDetector(confidence_threshold)
    
    # YOLO-NAS detectors
    elif model_type.startswith("yolo_nas"):
        size = model_type.split("_")[-1] if "_" in model_type and len(model_type.split("_")) > 2 else "l"
        return YOLONASDetector(confidence_threshold, model_size=size)
    
    # RT-DETR detectors
    elif model_type.startswith("rtdetr"):
        size = model_type.split("_")[-1] if "_" in model_type and len(model_type.split("_")) > 1 else "l"
        return RTDETRDetector(confidence_threshold, model_size=size)
    
    # Grounding DINO detectors
    elif model_type.startswith("grounding_dino"):
        size = model_type.split("_")[-1] if "_" in model_type and len(model_type.split("_")) > 2 else "b"
        return GroundingDINODetector(confidence_threshold, model_size=size)
    
    # Hybrid detector
    elif model_type == "hybrid":
        return HybridDetector(confidence_threshold)
    
    # Ensemble detector
    elif model_type == "ensemble":
        ensemble_method = kwargs.get("ensemble_method", "uncertainty")
        models = kwargs.get("models", None)
        return EnsembleDetector(confidence_threshold, ensemble_method=ensemble_method, models=models)
    
    # Detectron2 detectors
    elif model_type == "detectron2_fpn":
        return Detectron2Detector(confidence_threshold, model_name="mask_rcnn_R_50_FPN_1x")
    elif model_type == "detectron2_c4":
        return Detectron2Detector(confidence_threshold, model_name="mask_rcnn_R_50_C4_1x")
    
    # DeepLabV3+ detectors
    elif model_type == "deeplabv3_resnet101_ade20k":
        return DeepLabV3PlusDetector(confidence_threshold, model_name="deeplabv3_resnet101_ade20k")
    elif model_type == "deeplabv3_resnet50_ade20k":
        return DeepLabV3PlusDetector(confidence_threshold, model_name="deeplabv3_resnet50_ade20k")
    elif model_type == "deeplabv3_resnet101_voc":
        return DeepLabV3PlusDetector(confidence_threshold, model_name="deeplabv3_resnet101_voc")
    elif model_type == "deeplabv3_resnet50_voc":
        return DeepLabV3PlusDetector(confidence_threshold, model_name="deeplabv3_resnet50_voc")
    
    # Default to YOLOv8
    else:
        return StorageDetector(confidence_threshold)
