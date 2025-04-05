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

# Detection mode constants
DETECTION_MODE_FULL = "Full (Units & Compartments)"
DETECTION_MODE_UNITS_ONLY = "Units Only"
DETECTION_MODE_ALL_SEGMENTS = "All Segment"

def create_detector(model_type="yolo", confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, **kwargs):
    """
    Factory function to create the appropriate detector based on model type using a function dispatch pattern.
    
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
    
    # Function dispatch dictionary
    detector_factory = {
        "sam21": lambda: SAM21Detector(confidence_threshold),
        "fastsam": lambda: FastSAMDetector(confidence_threshold),
        "yolo_nas_s": lambda: YOLONASDetector(confidence_threshold, model_size="s"),
        "yolo_nas_m": lambda: YOLONASDetector(confidence_threshold, model_size="m"),
        "yolo_nas_l": lambda: YOLONASDetector(confidence_threshold, model_size="l"),
        "rtdetr_s": lambda: RTDETRDetector(confidence_threshold, model_size="s"),
        "rtdetr_m": lambda: RTDETRDetector(confidence_threshold, model_size="m"),
        "rtdetr_l": lambda: RTDETRDetector(confidence_threshold, model_size="l"),
        "rtdetr_x": lambda: RTDETRDetector(confidence_threshold, model_size="x"),
        "grounding_dino_t": lambda: GroundingDINODetector(confidence_threshold, model_size="t"),
        "grounding_dino_b": lambda: GroundingDINODetector(confidence_threshold, model_size="b"),
        "grounding_dino_l": lambda: GroundingDINODetector(confidence_threshold, model_size="l"),
        "hybrid": lambda: HybridDetector(confidence_threshold),
        "ensemble": lambda: EnsembleDetector(confidence_threshold, ensemble_method=kwargs.get("ensemble_method", "uncertainty"), models=kwargs.get("models", None)),
        "detectron2_fpn": lambda: Detectron2Detector(confidence_threshold, model_name="mask_rcnn_R_50_FPN_1x"),
        "detectron2_c4": lambda: Detectron2Detector(confidence_threshold, model_name="mask_rcnn_R_50_C4_1x"),
        "deeplabv3_resnet101_ade20k": lambda: DeepLabV3PlusDetector(confidence_threshold, model_name="deeplabv3_resnet101_ade20k"),
        "deeplabv3_resnet50_ade20k": lambda: DeepLabV3PlusDetector(confidence_threshold, model_name="deeplabv3_resnet50_ade20k"),
        "deeplabv3_resnet101_voc": lambda: DeepLabV3PlusDetector(confidence_threshold, model_name="deeplabv3_resnet101_voc"),
        "deeplabv3_resnet50_voc": lambda: DeepLabV3PlusDetector(confidence_threshold, model_name="deeplabv3_resnet50_voc")
    }
    
    # Create detector using dispatch
    if model_type in detector_factory:
        return detector_factory[model_type]()
    else:
        return StorageDetector(confidence_threshold)
