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
    DEFAULT_DETECTION_MODE,
    MODEL_OPTIONS
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
            options=MODEL_OPTIONS,
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
        elif "FastSAM" in model_type:
            st.info("FastSAM: Optimized version of SAM with faster processing but potentially less detail.")
        elif "Mask R-CNN (FPN)" in model_type:
            st.info("Mask R-CNN (FPN): Detectron2 model with Feature Pyramid Network for balanced performance and size (~170MB).")
        elif "Mask R-CNN (C4)" in model_type:
            st.info("Mask R-CNN (C4): Detectron2 model with ResNet-50-C4 backbone, smaller alternative (~160MB).")
        elif "DeepLabV3+ ResNet101 (ADE20K)" in model_type:
            st.info("DeepLabV3+ ResNet101 (ADE20K): Semantic segmentation model with ResNet101 backbone trained on ADE20K dataset. Optimized for Mac M1 with MPS support.")
        elif "DeepLabV3+ ResNet50 (ADE20K)" in model_type:
            st.info("DeepLabV3+ ResNet50 (ADE20K): Lighter semantic segmentation model with ResNet50 backbone trained on ADE20K dataset. Faster with good accuracy, optimized for Mac M1.")
        elif "DeepLabV3+ ResNet101 (VOC)" in model_type:
            st.info("DeepLabV3+ ResNet101 (VOC): Semantic segmentation model with ResNet101 backbone trained on PASCAL VOC dataset. Optimized for Mac M1 with MPS support.")
        elif "DeepLabV3+ ResNet50 (VOC)" in model_type:
            st.info("DeepLabV3+ ResNet50 (VOC): Lighter semantic segmentation model with ResNet50 backbone trained on PASCAL VOC dataset. Faster with good accuracy, optimized for Mac M1.")
        
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
            st.info("All Segment: Shows all raw segmentation output from the model without any filtering or categorization.")
        
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
                
                # Map UI model names to detector model types
                if "mask r-cnn (fpn)" in detector_model_type:
                    detector_model_type = "detectron2_fpn"
                elif "mask r-cnn (c4)" in detector_model_type:
                    detector_model_type = "detectron2_c4"
                elif "deeplabv3+ resnet101 (ade20k)" in detector_model_type:
                    detector_model_type = "deeplabv3_resnet101_ade20k"
                elif "deeplabv3+ resnet50 (ade20k)" in detector_model_type:
                    detector_model_type = "deeplabv3_resnet50_ade20k"
                elif "deeplabv3+ resnet101 (voc)" in detector_model_type:
                    detector_model_type = "deeplabv3_resnet101_voc"
                elif "deeplabv3+ resnet50 (voc)" in detector_model_type:
                    detector_model_type = "deeplabv3_resnet50_voc"
                
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
                    # For "All Segment" mode, we use a special method to get raw segmentation
                    # without any filtering or categorization into units and compartments
                    storage_units = detector.process_image_all_segments(
                        np.array(image)
                    )
                
                # Visualize results with the slider parameters
                annotated_image = visualize_segmentation(
                    np.array(image), 
                    storage_units,
                    epsilon_factor=epsilon_factor,
                    chaikin_iterations=chaikin_iterations
                )
                
                with col2:
                    if detection_mode == DETECTION_MODE_ALL_SEGMENTS:
                        st.subheader(f"Raw Segmentation Output ({model_type})")
                    else:
                        st.subheader(f"Detected Storage Units ({model_type})")
                    st.image(annotated_image, use_container_width=True)
                
                # Display results
                st.subheader("Detection Results")
                
                # Display detection summary
                total_units = len(storage_units)
                total_compartments = sum(len(unit.compartments) for unit in storage_units)
                
                # Create summary metrics
                col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                with col_metrics1:
                    st.metric("Total Storage Units", total_units)
                with col_metrics2:
                    st.metric("Total Compartments", total_compartments)
                with col_metrics3:
                    avg_confidence = 0
                    if total_units > 0:
                        avg_confidence = sum(unit.confidence for unit in storage_units) / total_units
                    st.metric("Average Confidence", f"{avg_confidence:.2f}")
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["Hierarchy View", "Detailed Table View", "Summary View"])
                
                with tab1:
                    # Display hierarchical tree view
                    hierarchy_html = create_hierarchy_tree(storage_units)
                    st.components.v1.html(hierarchy_html, height=400)
                
                with tab2:
                    # Create a more detailed table with all detected elements
                    rows = []
                    
                    for i, unit in enumerate(storage_units):
                        if detection_mode == DETECTION_MODE_ALL_SEGMENTS:
                            rows.append({
                                "ID": f"S{i+1}",
                                "Type": "Segment",
                                "Class": unit.class_name,
                                "Confidence": f"{unit.confidence:.2f}",
                                "Width": unit.width,
                                "Height": unit.height,
                                "Area (px²)": unit.area,
                                "Position": f"({unit.x1}, {unit.y1}) to ({unit.x2}, {unit.y2})",
                                "Parent": "None"
                            })
                        else:
                            rows.append({
                                "ID": f"U{i+1}",
                                "Type": "Storage Unit",
                                "Class": unit.class_name,
                                "Confidence": f"{unit.confidence:.2f}",
                                "Width": unit.width,
                                "Height": unit.height,
                                "Area (px²)": unit.area,
                                "Position": f"({unit.x1}, {unit.y1}) to ({unit.x2}, {unit.y2})",
                                "Compartments": len(unit.compartments),
                                "Parent": "None"
                            })
                            
                            for j, comp in enumerate(unit.compartments):
                                rows.append({
                                    "ID": f"U{i+1}-C{j+1}",
                                    "Type": "Compartment",
                                    "Class": comp.class_name,
                                    "Confidence": f"{comp.confidence:.2f}",
                                    "Width": comp.width,
                                    "Height": comp.height,
                                    "Area (px²)": comp.area,
                                    "Position": f"({comp.x1}, {comp.y1}) to ({comp.x2}, {comp.y2})",
                                    "Compartments": "",
                                    "Parent": f"U{i+1} ({unit.class_name})"
                                })
                    
                    if rows:
                        df = pd.DataFrame(rows)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No segments detected.")
                
                with tab3:
                    # Create a summary view with statistics by object type
                    if detection_mode != DETECTION_MODE_ALL_SEGMENTS:
                        # Collect unit types and their counts
                        unit_types = {}
                        compartment_types = {}
                        
                        for unit in storage_units:
                            unit_type = unit.class_name
                            if unit_type in unit_types:
                                unit_types[unit_type] += 1
                            else:
                                unit_types[unit_type] = 1
                            
                            for comp in unit.compartments:
                                comp_type = comp.class_name
                                if comp_type in compartment_types:
                                    compartment_types[comp_type] += 1
                                else:
                                    compartment_types[comp_type] = 1
                        
                        # Create summary tables
                        col_summary1, col_summary2 = st.columns(2)
                        
                        with col_summary1:
                            st.subheader("Storage Unit Types")
                            if unit_types:
                                unit_df = pd.DataFrame({
                                    "Type": list(unit_types.keys()),
                                    "Count": list(unit_types.values())
                                })
                                st.dataframe(unit_df, use_container_width=True)
                            else:
                                st.info("No storage units detected.")
                        
                        with col_summary2:
                            st.subheader("Compartment Types")
                            if compartment_types:
                                comp_df = pd.DataFrame({
                                    "Type": list(compartment_types.keys()),
                                    "Count": list(compartment_types.values())
                                })
                                st.dataframe(comp_df, use_container_width=True)
                            else:
                                st.info("No compartments detected.")
                    else:
                        # For all segments mode, just show segment types
                        segment_types = {}
                        for unit in storage_units:
                            segment_type = unit.class_name
                            if segment_type in segment_types:
                                segment_types[segment_type] += 1
                            else:
                                segment_types[segment_type] = 1
                        
                        st.subheader("Segment Types")
                        if segment_types:
                            segment_df = pd.DataFrame({
                                "Type": list(segment_types.keys()),
                                "Count": list(segment_types.values())
                            })
                            st.dataframe(segment_df, use_container_width=True)
                        else:
                            st.info("No segments detected.")
                
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
                - **Mask R-CNN (FPN)**: Detectron2 model with Feature Pyramid Network for balanced performance and size (~170MB).
                - **Mask R-CNN (C4)**: Detectron2 model with ResNet-50-C4 backbone, smaller alternative (~160MB).
                - **DeepLabV3+ ResNet101 (ADE20K)**: Semantic segmentation model with ResNet101 backbone trained on ADE20K dataset. Optimized for Mac M1 with MPS support.
                - **DeepLabV3+ ResNet50 (ADE20K)**: Lighter semantic segmentation model with ResNet50 backbone trained on ADE20K dataset. Faster with good accuracy, optimized for Mac M1.
                - **DeepLabV3+ ResNet101 (VOC)**: Semantic segmentation model with ResNet101 backbone trained on PASCAL VOC dataset. Optimized for Mac M1 with MPS support.
                - **DeepLabV3+ ResNet50 (VOC)**: Lighter semantic segmentation model with ResNet50 backbone trained on PASCAL VOC dataset. Faster with good accuracy, optimized for Mac M1.
                """)

if __name__ == "__main__":
    main()