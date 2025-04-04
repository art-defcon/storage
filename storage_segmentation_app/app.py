import streamlit as st
import os
import numpy as np
from PIL import Image
import pandas as pd

from detection import create_detector
from models import StorageUnit, StorageCompartment
from utils import visualize_segmentation, create_hierarchy_tree

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
            value=0.5, 
            step=0.05
        )
        
        # Model selection
        st.subheader("Model Selection")
        model_type = st.radio(
            "Segmentation Model",
            options=["YOLO", "SAM 2.1", "FastSAM"],
            index=0,
            help="Choose the segmentation model to use for detection"
        )
        
        # Model descriptions
        if model_type == "YOLO":
            st.info("YOLO: Fast and accurate object detection with good segmentation capabilities.")
        elif model_type == "SAM 2.1":
            st.info("SAM 2.1: State-of-the-art segmentation model with excellent detail, but slower processing.")
        else:  # FastSAM
            st.info("FastSAM: Optimized version of SAM with faster processing but potentially less detail.")
        
        detection_mode = st.radio(
            "Detection Mode",
            options=["Full (Units & Compartments)", "Units Only"]
        )
        
        process_button = st.button("Process Image")
    
    # Main content area
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        # Process image when button is clicked
        if process_button:
            with st.spinner(f"Processing image with {model_type}..."):
                # Convert model type to lowercase for detector creation
                detector_model_type = model_type.lower().replace(" ", "").replace(".", "")
                
                # Initialize detector with selected model
                detector = create_detector(
                    model_type=detector_model_type,
                    confidence_threshold=confidence_threshold
                )
                
                # Process image
                detect_compartments = detection_mode == "Full (Units & Compartments)"
                storage_units = detector.process_image(
                    np.array(image), 
                    detect_compartments=detect_compartments
                )
                
                # Visualize results
                annotated_image = visualize_segmentation(
                    np.array(image), 
                    storage_units
                )
                
                with col2:
                    st.subheader(f"Detected Storage Units ({model_type})")
                    st.image(annotated_image, use_column_width=True)
                
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
                - **SAM 2.1**: State-of-the-art segmentation with excellent boundary precision. Best for detailed analysis.
                - **FastSAM**: Optimized for speed while maintaining good segmentation quality. Good balance of speed and accuracy.
                """)

if __name__ == "__main__":
    main()