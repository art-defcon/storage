#!/usr/bin/env python3
"""
Example script demonstrating how to use the Storage Segmentation functionality
programmatically without the Streamlit interface.
"""

import cv2
import numpy as np
import argparse
import json
import os
from pathlib import Path

from detection import StorageDetector
from utils import visualize_segmentation, export_results_to_json

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Storage Segmentation Example')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save output files')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--compartments', action='store_true', help='Detect compartments within units')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load image
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return
    
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    print(f"Processing image: {image_path}")
    print(f"Image dimensions: {image.shape[1]}x{image.shape[0]}")
    
    # Initialize detector
    detector = StorageDetector(confidence_threshold=args.confidence)
    
    # Process image
    print("Detecting storage units...")
    storage_units = detector.process_image(
        image,
        detect_compartments=args.compartments
    )
    
    # Print detection results
    print(f"Found {len(storage_units)} storage units")
    for i, unit in enumerate(storage_units):
        print(f"  Unit {i+1}: {unit.class_name} ({unit.confidence:.2f}) - {unit.width}x{unit.height}")
        print(f"    Compartments: {len(unit.compartments)}")
        for j, comp in enumerate(unit.compartments):
            print(f"      Compartment {j+1}: {comp.class_name} ({comp.confidence:.2f}) - {comp.width}x{comp.height}")
    
    # Visualize results
    print("Generating visualization...")
    annotated_image = visualize_segmentation(
        image,
        storage_units,
        show_labels=True
    )
    
    # Save results
    output_base = image_path.stem
    
    # Save annotated image
    output_image_path = Path(args.output_dir) / f"{output_base}_annotated.jpg"
    cv2.imwrite(str(output_image_path), annotated_image)
    print(f"Saved annotated image to {output_image_path}")
    
    # Export results to JSON
    results = export_results_to_json(storage_units)
    output_json_path = Path(args.output_dir) / f"{output_base}_results.json"
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {output_json_path}")

if __name__ == "__main__":
    main()