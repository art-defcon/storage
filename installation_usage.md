# Installation and Usage Guide

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd storage-segmentation-app
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open my web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. Upload an image of my storage furniture using the file uploader in the sidebar

4. Adjust the detection settings to suit my needs:
   - Confidence Threshold: Set the minimum confidence score for my detections
   - Object Size: Filter objects based on percentage of my screen width (ranges from 1% to 50%)
   - Contour Smoothing Factor: Adjust the smoothness of the contours
   - Corner Rounding Iterations: Adjust the roundness of the corners
   - Segmentation Model: Choose from multiple models including:
     - YOLO models (YOLO, YOLO-NAS S/M/L)
     - Transformer models (RT-DETR S/M/L/X)
     - Segmentation models (SAM 2.1 tiny/small/base, FastSAM)
     - Vision-language models (Grounding DINO)
     - Advanced pipelines (Hybrid Pipeline, Ensemble methods)
     - Detectron2 models (Mask R-CNN FPN/C4)
     - DeepLabV3+ models (ResNet50/101 with ADE20K/VOC)
   - Detection Mode: Choose between:
     - Full (Units & Compartments): Detects both my storage units and their internal compartments
     - Units Only: Detects only my main storage units without internal compartments
     - All Segment: Shows all segmentation without filtering any objects
   - Dynamic Confidence Thresholds: Enable furniture type-specific confidence thresholds

5. For Ensemble models, I can configure:
   - Ensemble Method: uncertainty, average, max, or vote
   - Models to include in the ensemble
   - Test-Time Augmentation toggle

6. Click the "Process Image" button to start the detection

7. View my results:
   - Original and annotated images side by side
   - Hierarchy view showing the structure of my detected storage units and compartments
   - Detailed table view with information about each detected element
   - Summary view with statistics by object type
   - Advanced info tab with metadata and model-specific information

## Object Size Filtering

The "Object size" slider lets me filter detected objects based on their width relative to my image width:

- The slider ranges from 1% to 50% of my image width, with a default value of 20%
- Objects smaller than my selected percentage will be filtered out
- This filtering applies to all detection modes:
  - In "Full" mode, both units and compartments smaller than my threshold will be filtered
  - In "Units Only" mode, only units smaller than my threshold will be filtered
  - In "All Segment" mode, all detected segments smaller than my threshold will be filtered

This feature is particularly useful for:
- Removing small, irrelevant objects from my detection results
- Focusing on larger storage units when analyzing my complex scenes
- Adjusting detection sensitivity based on my specific image content

## Customization

I can customize the application by:

- Selecting different models based on my specific needs:
  - YOLO for faster processing
  - SAM 2.1 for more precise segmentation
  - YOLO-NAS for higher accuracy
  - RT-DETR for transformer-based detection
  - Grounding DINO for text-prompted detection
- Adjusting the visualization parameters in the UI:
  - Contour smoothing for cleaner outlines
  - Corner rounding for more natural furniture representation
- Configuring ensemble methods and included models
- Enabling or disabling dynamic confidence thresholds
- Adjusting object size filtering for my specific furniture