# Storage Segmentation App

This application uses AI/ML to scan images of storage furniture, identify different storage types, and segment individual compartments within each storage unit. It creates a structured list of all available storage spaces and displays the image with visual annotations showing all identified segments.

## Features

- Scan images of storage furniture
- Identify different storage types using YOLO11x-seg model for improved accuracy
- Segment individual compartments within each storage unit
- Create a structured list of all available storage spaces
- Display the image with visual annotations showing all identified segments

## Technical Requirements

- Python 3.8+
- Dependencies: ultralytics (YOLO), OpenCV, Streamlit, NumPy, pandas, Pillow

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

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. Upload an image of storage furniture using the file uploader in the sidebar

4. Adjust the detection settings as needed:
   - Confidence Threshold: Minimum confidence score for detections
   - Detection Mode: Choose between:
     - Full (Units & Compartments): Detects both storage units and their internal compartments
     - Units Only: Detects only the main storage units without internal compartments
     - All Segment: Shows all segmentation without filtering any objects
   - Object Size: Filter objects based on percentage of screen width (ranges from 1% to 50% with a default of 15%)

5. Click the "Process Image" button to start the detection

6. View the results:
   - Original and annotated images side by side
   - Hierarchy view showing the structure of detected storage units and compartments
   - Table view with detailed information about each detected element

## Object Size Filtering

The "Object size" slider allows you to filter detected objects based on their width relative to the image width:

- The slider ranges from 1% to 50% of the image width, with a default value of 15%
- Objects smaller than the selected percentage will be filtered out
- This filtering applies to all detection modes:
  - In "Full" mode, both units and compartments smaller than the threshold will be filtered
  - In "Units Only" mode, only units smaller than the threshold will be filtered
  - In "All Segment" mode, all detected segments smaller than the threshold will be filtered

This feature is particularly useful for:
- Removing small, irrelevant objects from the detection results
- Focusing on larger storage units when analyzing complex scenes
- Adjusting detection sensitivity based on the specific image content

## Project Structure

- `app.py`: Main Streamlit application
- `detection.py`: Image processing pipeline using YOLO11x-seg
- `models.py`: Data structures for storage units and compartments
- `utils.py`: Visualization and helper functions
- `requirements.txt`: List of required dependencies
- `data/models/`: Directory for storing YOLO models, including yolo11x-seg.pt

## How It Works

The application uses a two-stage detection pipeline:

1. First-level segmentation for storage unit detection
2. Second-level segmentation for compartment detection within each unit

The YOLO11x-seg model is used for both stages, providing improved segmentation accuracy compared to previous versions. The second stage focuses on the cropped regions of detected storage units.

## Model Information

The application now uses the YOLO11x-seg model, which offers several advantages over the previous YOLOv8 model:
- Improved segmentation accuracy for better boundary detection
- Enhanced feature extraction capabilities
- Better performance on complex storage furniture with multiple compartments
- More precise mask generation for irregular shapes

## Libraries Used

This project relies on several powerful open-source libraries:

- **Ultralytics YOLO**: State-of-the-art object detection and segmentation framework that powers the core detection capabilities. [Documentation](https://docs.ultralytics.com/)
- **Streamlit**: Interactive web application framework for creating data apps with minimal code. [Documentation](https://docs.streamlit.io/)
- **OpenCV**: Computer vision library used for image processing and manipulation. [Documentation](https://docs.opencv.org/)
- **NumPy**: Fundamental package for scientific computing with Python, used for array operations and numerical processing. [Documentation](https://numpy.org/doc/)
- **pandas**: Data analysis and manipulation library, used for structured data handling. [Documentation](https://pandas.pydata.org/docs/)
- **Pillow**: Python Imaging Library fork, used for image opening, manipulation, and saving. [Documentation](https://pillow.readthedocs.io/)

## Customization

You can customize the application by:

- Training custom YOLO models for specific types of storage furniture
- Adjusting the visualization parameters in the `utils.py` file
- Adding additional metadata fields to the storage unit and compartment classes

## Credits and Acknowledgements

This project is based on the Ultralytics YOLO implementation by Glenn Jocher and the Ultralytics team. The YOLO11x-seg model used in this application is developed and maintained by Ultralytics.

- **Ultralytics YOLO**: [GitHub Repository](https://github.com/ultralytics/ultralytics)
- **YOLO11x-seg Model**: [Model Documentation](https://docs.ultralytics.com/tasks/segment/)
- **Author**: John Petroff
- **Project Repository**: [GitHub](https://github.com/johnpetroff/storage-segmentation-app)
