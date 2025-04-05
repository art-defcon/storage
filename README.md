# Storage Segmentation App

This application uses advanced AI/ML models to scan images of my storage furniture, identify different storage types, and segment individual compartments within each storage unit. It creates a structured list of all available storage spaces and displays the image with visual annotations showing all identified segments.

The purpose for creating this app was to get a sence of accuracy in different models for segmentation and also to test pretrainded weights in each model.ß 

![Storage Segmentation App Screenshot](https://github.com/art-defcon/storage/blob/main/public/screenshot_updated_2.png)
*The Storage Segmentation App interface showing multiple detection models, ensemble methods, and visualization options that help me customize how storage units are identified and displayed.*

## Features

- Scan images of my storage furniture
- Multiple detection models to choose from:
  - YOLO for fast detection
  - YOLO-NAS with 10-17% higher mAP than YOLOv8
  - RT-DETR (Real-Time Detection Transformer) combining transformer accuracy with YOLO speed
  - SAM 2.1 for precise segmentation
  - FastSAM for optimized segmentation
  - Grounding DINO for vision-language capabilities and zero-shot detection
  - Mask R-CNN models from Detectron2
  - DeepLabV3+ models optimized for Mac M1
- Advanced detection pipelines:
  - Hybrid Pipeline combining YOLO-NAS, Grounding DINO, and SAM 2.1
  - Ensemble methods (uncertainty-aware, average, max, vote)
- Segment individual compartments within each storage unit
- Create a structured list of all my available storage spaces
- Display my image with visual annotations showing all identified segments
- Hierarchical classification with parent-child relationships
- Dynamic confidence thresholds based on furniture type
- Enhanced visualization with contour smoothing and corner rounding

## Technical Requirements

- Python 3.8+
- Dependencies:
  - ultralytics (YOLO, SAM 2.1, FastSAM)
  - torch and torchvision
  - groundingdino-py
  - huggingface-hub
  - supervision
  - timm
  - transformers
  - streamlit
  - OpenCV
  - NumPy
  - pandas
  - Pillow
  - scikit-learn
  - matplotlib

For installation and usage instructions, please refer to the [Installation and Usage Guide](installation_usage.md).

## Project Structure

- `app.py`: Main Streamlit application
- `config.py`: Configuration settings for the application
- `utils.py`: Visualization and helper functions
- `models.py`: Data structures for storage units and compartments
- `requirements.txt`: List of required dependencies
- `detectors/`: Directory containing all detector implementations:
  - `base_detector.py`: Base class for all detectors
  - `yolo_detector.py`: YOLOv8 detector implementation
  - `yolo_nas_detector.py`: YOLO-NAS detector implementation
  - `rt_detr_detector.py`: RT-DETR detector implementation
  - `sam_detector.py`: SAM 2.1 detector implementation
  - `fastsam_detector.py`: FastSAM detector implementation
  - `grounding_dino_detector.py`: Grounding DINO detector implementation
  - `detectron2_detector.py`: Detectron2 Mask R-CNN detector implementation
  - `deeplabv3_detector.py`: DeepLabV3+ detector implementation
  - `hybrid_detector.py`: Hybrid pipeline implementation
  - `ensemble_detector.py`: Ensemble methods implementation
  - `factory.py`: Factory for creating detector instances
- `data/models/`: Directory for storing models
- `data/samples/`: Sample images for testing

## How It Works

The application now uses a multi-model approach with several detection pipelines:

1. **Single Model Pipeline**:
   - Uses a single model (YOLO, YOLO-NAS, RT-DETR, SAM 2.1, etc.) for both detection and segmentation
   - Configurable for different model sizes and confidence thresholds

2. **Hybrid Pipeline**:
   - YOLO-NAS for initial furniture unit detection
   - Grounding DINO for component classification using text prompts
   - SAM 2.1 for precise segmentation of the detected units

3. **Ensemble Pipeline**:
   - Combines predictions from multiple models using different strategies:
     - Uncertainty-aware ensemble: Weights predictions by uncertainty estimates
     - Average ensemble: Averages predictions from all models
     - Max ensemble: Takes the maximum confidence prediction
     - Vote ensemble: Uses majority voting for final predictions
   - Supports test-time augmentation for improved accuracy

All pipelines support hierarchical classification, with storage units containing compartments, and dynamic confidence thresholds based on furniture type.

## Libraries Used

This project relies on several powerful open-source libraries:

- **Ultralytics YOLO**: State-of-the-art object detection and segmentation framework that powers the core detection capabilities. [Documentation](https://docs.ultralytics.com/)
- **YOLO-NAS**: Neural Architecture Search version of YOLO with 10-17% higher mAP. [Documentation](https://docs.ultralytics.com/models/yolo-nas/)
- **RT-DETR**: Real-Time Detection Transformer combining transformer accuracy with YOLO speed. [Documentation](https://docs.ultralytics.com/models/rtdetr/)
- **SAM 2.1**: Segment Anything Model for precise segmentation. [Documentation](https://docs.ultralytics.com/models/sam/)
- **Grounding DINO**: Vision-language model for zero-shot detection. [GitHub](https://github.com/IDEA-Research/GroundingDINO)
- **Detectron2**: Facebook AI Research's detection and segmentation framework. [Documentation](https://detectron2.readthedocs.io/)
- **Streamlit**: Interactive web application framework for creating data apps with minimal code. [Documentation](https://docs.streamlit.io/)
- **OpenCV**: Computer vision library used for image processing and manipulation. [Documentation](https://docs.opencv.org/)
- **NumPy**: Fundamental package for scientific computing with Python, used for array operations and numerical processing. [Documentation](https://numpy.org/doc/)
- **pandas**: Data analysis and manipulation library, used for structured data handling. [Documentation](https://pandas.pydata.org/docs/)
- **Pillow**: Python Imaging Library fork, used for image opening, manipulation, and saving. [Documentation](https://pillow.readthedocs.io/)
- **PyTorch**: Deep learning framework that powers all the models. [Documentation](https://pytorch.org/docs/)
- **Hugging Face**: Platform for sharing and using machine learning models. [Documentation](https://huggingface.co/docs)
- **Supervision**: Computer vision annotation toolkit. [GitHub](https://github.com/roboflow/supervision)
