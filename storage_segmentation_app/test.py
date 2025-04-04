#!/usr/bin/env python3
"""
Test script for the Storage Segmentation App.
This script tests the core functionality of the application.
"""

import unittest
import os
import numpy as np
import cv2
from pathlib import Path

from detection import StorageDetector
from models import StorageUnit, StorageCompartment
from utils import visualize_segmentation, export_results_to_json

class TestStorageSegmentation(unittest.TestCase):
    """Test cases for the Storage Segmentation App."""
    
    def setUp(self):
        """Set up test environment."""
        # Create test data directory if it doesn't exist
        self.test_data_dir = Path("data/test")
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Create a simple test image (a white rectangle on black background)
        self.test_image = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.rectangle(self.test_image, (100, 100), (400, 400), (255, 255, 255), -1)
        
        # Save the test image
        self.test_image_path = self.test_data_dir / "test_image.jpg"
        cv2.imwrite(str(self.test_image_path), self.test_image)
        
        # Create a mock storage unit for testing
        self.mock_unit = StorageUnit(
            x1=100, y1=100, x2=400, y2=400,
            confidence=0.9,
            class_id=0,
            class_name="Bookshelf",
            mask=None
        )
        
        # Add mock compartments
        self.mock_unit.add_compartment(
            StorageCompartment(
                x1=120, y1=120, x2=380, y2=200,
                confidence=0.85,
                class_id=0,
                class_name="Shelf",
                mask=None,
                parent_unit=self.mock_unit
            )
        )
        
        self.mock_unit.add_compartment(
            StorageCompartment(
                x1=120, y1=220, x2=380, y2=300,
                confidence=0.8,
                class_id=0,
                class_name="Shelf",
                mask=None,
                parent_unit=self.mock_unit
            )
        )
        
        self.mock_unit.add_compartment(
            StorageCompartment(
                x1=120, y1=320, x2=380, y2=380,
                confidence=0.75,
                class_id=0,
                class_name="Drawer",
                mask=None,
                parent_unit=self.mock_unit
            )
        )
    
    def test_detector_initialization(self):
        """Test that the detector can be initialized."""
        detector = StorageDetector(confidence_threshold=0.5)
        self.assertIsNotNone(detector)
        self.assertEqual(detector.confidence_threshold, 0.5)
    
    def test_model_properties(self):
        """Test the properties of the storage unit and compartment models."""
        # Test storage unit properties
        self.assertEqual(self.mock_unit.width, 300)
        self.assertEqual(self.mock_unit.height, 300)
        self.assertEqual(self.mock_unit.area, 90000)
        self.assertEqual(self.mock_unit.center, (250, 250))
        self.assertEqual(self.mock_unit.compartment_count, 3)
        
        # Test compartment properties
        shelf_compartments = self.mock_unit.get_compartments_by_class("Shelf")
        self.assertEqual(len(shelf_compartments), 2)
        
        drawer_compartments = self.mock_unit.get_compartments_by_class("Drawer")
        self.assertEqual(len(drawer_compartments), 1)
    
    def test_visualization(self):
        """Test the visualization function."""
        # Create a visualization of the mock unit
        annotated_image = visualize_segmentation(
            self.test_image,
            [self.mock_unit],
            show_labels=True
        )
        
        # Check that the result is a valid image
        self.assertIsInstance(annotated_image, np.ndarray)
        self.assertEqual(annotated_image.shape, self.test_image.shape)
        
        # Save the annotated image for manual inspection
        output_path = self.test_data_dir / "test_annotated.jpg"
        cv2.imwrite(str(output_path), annotated_image)
    
    def test_export_to_json(self):
        """Test exporting results to JSON."""
        # Export the mock unit to JSON
        results = export_results_to_json([self.mock_unit])
        
        # Check the structure of the results
        self.assertIn("units_count", results)
        self.assertEqual(results["units_count"], 1)
        self.assertIn("units", results)
        self.assertEqual(len(results["units"]), 1)
        
        # Check the unit data
        unit_data = results["units"][0]
        self.assertEqual(unit_data["class_name"], "Bookshelf")
        self.assertEqual(unit_data["compartment_count"], 3)
        self.assertIn("compartments", unit_data)
        self.assertEqual(len(unit_data["compartments"]), 3)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test files
        if self.test_image_path.exists():
            os.remove(self.test_image_path)
        
        test_annotated_path = self.test_data_dir / "test_annotated.jpg"
        if test_annotated_path.exists():
            os.remove(test_annotated_path)

if __name__ == "__main__":
    unittest.main()