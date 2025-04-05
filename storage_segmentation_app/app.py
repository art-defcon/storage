import streamlit as st
import os
import numpy as np
from PIL import Image
import pandas as pd

from detection import create_detector, DETECTION_MODE_FULL, DETECTION_MODE_UNITS_ONLY, DETECTION_MODE_ALL_SEGMENTS
from models import StorageUnit, StorageCompartment
from utils import visualize_segmentation, create_hierarchy_tree
from config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_OBJECT_SIZE,
    DEFAULT_CONTOUR_SMOOTHING_FACTOR,
    DEFAULT_CORNER_ROUNDING_ITERATIONS,
    DEFAULT_SEGMENTATION_MODEL,
    DEFAULT_MODEL_INDEX,
    DEFAULT_DETECTION_MODE
)

# Set page configuration
st.set_page_config(
    page_title="Storage Segmentation App",
    page_icon="📦",
    layout="wide"
)

def main():
    # App title and description
    st.title("Storage Segmentation App")
    st.markdown("""
    This application uses AI to detect storage furniture and segment individual compartments.
    It supports multiple segmentation models for different use cases.
    Upload an image of storage furniture to get started.
    """)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Settings")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        # Detection settings
        st.subheader("Detection Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.1, 
            max_value=1.0, 
            value=DEFAULT_CONFIDENCE_THRESHOLD, 
            step=0.05
        )
        
        # Object size filter slider
        object_size_percentage = st.slider(
            "Object size", 
            min_value=1, 
            max_value=50, 
            value=DEFAULT_OBJECT_SIZE, 
            step=1,
            help="Filter out objects smaller than this percentage of the image width"
        )
        
        # Visualization settings
        st.subheader("Visualization Settings")
        
        # Add sliders for epsilon_factor and chaikin_iterations
        epsilon_factor = st.slider(
            "Contour Smoothing Factor", 
            min_value=0.001, 
            max_value=0.1, 
            value=DEFAULT_CONTOUR_SMOOTHING_FACTOR, 
            step=0.001,
            help="Higher values result in smoother contours (Douglas-Peucker algorithm)"
        )
        
        chaikin_iterations = st.slider(
            "Corner Rounding Iterations", 
            min_value=0, 
            max_value=5, 
            value=DEFAULT_CORNER_ROUNDING_ITERATIONS, 
            step=1,
            help="Higher values result in more rounded corners (Chaikin's algorithm)"
        )
        
        # Model selection
        st.subheader("Model Selection")
        model_type = st.radio(
            "Segmentation Model",
            options=["YOLO", "SAM 2.1 tiny", "SAM 2.1 small", "SAM 2.1 base ⚠️", "FastSAM"],
            index=DEFAULT_MODEL_INDEX,
            help="Choose the segmentation model to use for detection"
        )
        
        # Model descriptions
        if model_type == "YOLO":
            st.info("YOLO: Fast and accurate object detection with good segmentation capabilities.")
        elif "SAM 2.1 tiny" in model_type:
            st.info("SAM 2.1 tiny: Lightweight version of SAM 2.1 with faster processing and good segmentation quality.")
        elif "SAM 2.1 small" in model_type:
            st.info("SAM 2.1 small: Medium-sized SAM 2.1 model with balanced performance and accuracy.")
        elif "SAM 2.1 base" in model_type:
            st.warning("SAM 2.1 base: Full-sized SAM 2.1 model with excellent detail and precision, but slower processing time.")
        else:  # FastSAM
            st.info("FastSAM: Optimized version of SAM with faster processing but potentially less detail.")
        
        detection_mode = st.radio(
            "Detection Mode",
            options=[
                DETECTION_MODE_FULL, 
                DETECTION_MODE_UNITS_ONLY,
                DETECTION_MODE_ALL_SEGMENTS
            ],
            index=[DETECTION_MODE_FULL, DETECTION_MODE_UNITS_ONLY, DETECTION_MODE_ALL_SEGMENTS].index(DEFAULT_DETECTION_MODE)
        )
        
        # Detection mode descriptions
        if detection_mode == DETECTION_MODE_FULL:
            st.info("Full: Detects both storage units and their compartments.")
        elif detection_mode == DETECTION_MODE_UNITS_ONLY:
            st.info("Units Only: Detects only storage units without their compartments.")
        elif detection_mode == DETECTION_MODE_ALL_SEGMENTS:
            st.info("All Segment: Shows all segmentation without filtering any objects.")
        
        process_button = st.button("Process Image")
    
    # Main content area
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        # Process image when button is clicked
        if process_button:
            with st.spinner(f"Processing image with {model_type}..."):
                # Convert model type to lowercase for detector creation
                detector_model_type = model_type.lower().replace(" ", "").replace(".", "").replace("⚠️", "")
                
                # Initialize detector with selected model
                detector = create_detector(
                    model_type=detector_model_type,
                    confidence_threshold=confidence_threshold
                )
                
                # Calculate min_segment_width based on the object_size_percentage slider
                image_width = image.width
                min_segment_width = int(image_width * (object_size_percentage / 100))
                
                # Process image based on detection mode
                if detection_mode == DETECTION_MODE_FULL:
                    storage_units = detector.process_image(
                        np.array(image), 
                        detect_compartments=True,
                        filter_small_segments=True,
                        min_segment_width=min_segment_width
                    )
                elif detection_mode == DETECTION_MODE_UNITS_ONLY:
                    storage_units = detector.process_image(
                        np.array(image), 
                        detect_compartments=False,
                        filter_small_segments=True,
                        min_segment_width=min_segment_width
                    )
                elif detection_mode == DETECTION_MODE_ALL_SEGMENTS:
                    # For "All Segment" mode, we detect all segments without filtering
                    storage_units = detector.process_image(
                        np.array(image), 
                        detect_compartments=True,
                        filter_small_segments=False
                    )
                
                # Visualize results with the slider parameters
                annotated_image = visualize_segmentation(
                    np.array(image), 
                    storage_units,
                    epsilon_factor=epsilon_factor,
                    chaikin_iterations=chaikin_iterations
                )
                
                with col2:
                    st.subheader(f"Detected Storage Units ({model_type})")
                    st.image(annotated_image, use_container_width=True)
                
                # Display results
                st.subheader("Detection Results")
                
                # Create tabs for different views
                tab1, tab2 = st.tabs(["Hierarchy View", "Table View"])
                
                with tab1:
                    # Display hierarchical tree view
                    hierarchy_html = create_hierarchy_tree(storage_units)
                    st.components.v1.html(hierarchy_html, height=400)
                
                with tab2:
                    # Create a table with all detected elements
                    rows = []
                    
                    for unit in storage_units:
                        rows.append({
                            "Type": "Storage Unit",
                            "Class": unit.class_name,
                            "Confidence": f"{unit.confidence:.2f}",
                            "Dimensions": f"{unit.width}x{unit.height}",
                            "Parent": "None"
                        })
                        
                        for comp in unit.compartments:
                            rows.append({
                                "Type": "Compartment",
                                "Class": comp.class_name,
                                "Confidence": f"{comp.confidence:.2f}",
                                "Dimensions": f"{comp.width}x{comp.height}",
                                "Parent": unit.class_name
                            })
                    
                    if rows:
                        df = pd.DataFrame(rows)
                        st.dataframe(df)
                    else:
                        st.info("No storage units detected.")
                
                # Model comparison information
                st.subheader("Model Information")
                st.markdown(f"""
                **Current Model: {model_type}**
                
                **Model Comparison:**
                - **YOLO**: Fast detection with good accuracy for common objects. Best for real-time applications.
                - **SAM 2.1 tiny**: Lightweight segmentation model with good speed and reasonable accuracy.
                - **SAM 2.1 small**: Medium-sized model with balanced performance and accuracy.
                - **SAM 2.1 base** ⚠️: Full-sized model with excellent boundary precision. Best for detailed analysis but slower.
                - **FastSAM**: Optimized for speed while maintaining good segmentation quality. Good balance of speed and accuracy.
                """)

if __name__ == "__main__":
    main()