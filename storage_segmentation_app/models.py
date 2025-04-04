import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

class BoundingBox:
    """Base class for objects with bounding boxes and detection metadata."""
    
    def __init__(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        confidence: float,
        class_id: int,
        class_name: str,
        mask: Optional[np.ndarray] = None
    ):
        """
        Initialize a bounding box object.
        
        Args:
            x1, y1: Top-left coordinates
            x2, y2: Bottom-right coordinates
            confidence: Detection confidence score
            class_id: Class ID from the model
            class_name: Human-readable class name
            mask: Segmentation mask (optional)
        """
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.mask = mask
        
    @property
    def width(self) -> int:
        """Get the width of the bounding box."""
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        """Get the height of the bounding box."""
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        """Get the area of the bounding box."""
        return self.width * self.height
    
    @property
    def center(self) -> tuple:
        """Get the center coordinates of the bounding box."""
        return (
            self.x1 + self.width // 2,
            self.y1 + self.height // 2
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the object to a dictionary."""
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "has_mask": self.mask is not None
        }


class StorageCompartment(BoundingBox):
    """Class representing a storage compartment within a storage unit."""
    
    def __init__(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        confidence: float,
        class_id: int,
        class_name: str,
        mask: Optional[np.ndarray] = None,
        parent_unit: Optional['StorageUnit'] = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Initialize a storage compartment.
        
        Args:
            x1, y1, x2, y2: Bounding box coordinates
            confidence: Detection confidence score
            class_id: Class ID from the model
            class_name: Human-readable class name
            mask: Segmentation mask (optional)
            parent_unit: Reference to the parent storage unit
            metadata: Additional metadata about the compartment
        """
        super().__init__(x1, y1, x2, y2, confidence, class_id, class_name, mask)
        self.parent_unit = parent_unit
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the compartment to a dictionary."""
        result = super().to_dict()
        result.update({
            "parent_unit_id": id(self.parent_unit) if self.parent_unit else None,
            "parent_unit_class": self.parent_unit.class_name if self.parent_unit else None,
            "metadata": self.metadata
        })
        return result


class StorageUnit(BoundingBox):
    """Class representing a storage unit with multiple compartments."""
    
    def __init__(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        confidence: float,
        class_id: int,
        class_name: str,
        mask: Optional[np.ndarray] = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Initialize a storage unit.
        
        Args:
            x1, y1, x2, y2: Bounding box coordinates
            confidence: Detection confidence score
            class_id: Class ID from the model
            class_name: Human-readable class name
            mask: Segmentation mask (optional)
            metadata: Additional metadata about the unit
        """
        super().__init__(x1, y1, x2, y2, confidence, class_id, class_name, mask)
        self.compartments: List[StorageCompartment] = []
        self.metadata = metadata or {}
    
    def add_compartment(self, compartment: StorageCompartment) -> None:
        """Add a compartment to this storage unit."""
        compartment.parent_unit = self
        self.compartments.append(compartment)
    
    def get_compartments_by_class(self, class_name: str) -> List[StorageCompartment]:
        """Get all compartments of a specific class."""
        return [c for c in self.compartments if c.class_name == class_name]
    
    @property
    def compartment_count(self) -> int:
        """Get the number of compartments in this unit."""
        return len(self.compartments)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the storage unit to a dictionary."""
        result = super().to_dict()
        result.update({
            "compartment_count": self.compartment_count,
            "compartments": [c.to_dict() for c in self.compartments],
            "metadata": self.metadata
        })
        return result