import streamlit as st
import os
import numpy as np
from PIL import Image
import pandas as pd

from detection import StorageDetector
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
    It utilizes the YOLO11x-seg model for improved segmentation accuracy.
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
            with st.spinner("Processing image..."):
                # Initialize detector
                detector = StorageDetector(confidence_threshold=confidence_threshold)
                
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
                    st.subheader("Detected Storage Units")
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

if __name__ == "__main__":
    main()