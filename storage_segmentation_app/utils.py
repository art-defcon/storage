import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import random
import colorsys
from typing import List, Dict, Any, Tuple

from models import StorageUnit, StorageCompartment

def darken_color(color, factor=0.7):
    """
    Create a darker shade of a color.
    
    Args:
        color: Tuple of (R, G, B) values
        factor: Darkening factor (lower = darker)
        
    Returns:
        Tuple of darkened (R, G, B) values
    """
    r, g, b = color
    return (int(r * factor), int(g * factor), int(b * factor))

def calculate_line_width(image_width, factor=0.003, min_width=1):
    """
    Calculate line width as a percentage of image width.
    
    Args:
        image_width: Width of the image in pixels
        factor: Percentage factor (0.003 = 0.3%)
        min_width: Minimum width in pixels
        
    Returns:
        Line width in pixels
    """
    return max(min_width, int(image_width * factor))

def chaikin_corner_cutting(points, iterations=3):
    """
    Apply Chaikin's corner cutting algorithm for smooth curves with rounded corners.
    
    Args:
        points: List of points [(x1,y1), (x2,y2), ...]
        iterations: Number of iterations to perform
        
    Returns:
        Smoothed points with rounded corners
    """
    if len(points) < 3:
        return points
        
    for _ in range(iterations):
        new_points = []
        # Process all points except first and last
        new_points.append(points[0])  # Keep first point
        
        for i in range(len(points) - 1):
            p0 = points[i]
            p1 = points[i + 1]
            
            # Calculate 1/4 and 3/4 points
            q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
            r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
            
            new_points.append(q)
            new_points.append(r)
        
        new_points.append(points[-1])  # Keep last point
        points = new_points
        
    return points

def smooth_contour(contour, epsilon_factor=0.02, apply_chaikin=True, chaikin_iterations=3):
    """
    Smooth a contour using Douglas-Peucker algorithm and Chaikin's corner cutting.
    
    Args:
        contour: OpenCV contour
        epsilon_factor: Smoothing factor (higher = more smoothing)
        apply_chaikin: Whether to apply Chaikin's corner cutting
        chaikin_iterations: Number of Chaikin iterations
        
    Returns:
        Smoothed contour
    """
    # Apply Douglas-Peucker algorithm to reduce points
    perimeter = cv2.arcLength(contour, True)
    epsilon = epsilon_factor * perimeter
    approx_contour = cv2.approxPolyDP(contour, epsilon, True)
    
    if not apply_chaikin or len(approx_contour) < 3:
        return approx_contour
    
    # Convert to list of points for Chaikin's algorithm
    points = [tuple(point[0]) for point in approx_contour]
    
    # Apply Chaikin's corner cutting for rounded corners
    smoothed_points = chaikin_corner_cutting(points, iterations=chaikin_iterations)
    
    # Convert back to OpenCV contour format
    return np.array(smoothed_points).reshape(-1, 1, 2).astype(np.int32)

def generate_colors(n: int) -> List[Tuple[int, int, int]]:
    """
    Generate n visually distinct colors.
    
    Args:
        n: Number of colors to generate
        
    Returns:
        List of RGB color tuples
    """
    colors = []
    for i in range(n):
        # Use HSV color space for better visual distinction
        h = i / n
        s = 0.8
        v = 0.9
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors

def visualize_segmentation(
    image: np.ndarray,
    storage_units: List[StorageUnit],
    alpha: float = 0.3,
    show_labels: bool = True,
    epsilon_factor: float = 0.02,
    chaikin_iterations: int = 2
) -> np.ndarray:
    """
    Visualize detected storage units and compartments on the image using segmentation masks.
    
    Args:
        image: Input image
        storage_units: List of detected storage units
        alpha: Transparency of the overlay
        show_labels: Whether to show labels
        epsilon_factor: Smoothing factor for contours (higher = more smoothing)
        chaikin_iterations: Number of iterations for Chaikin's corner cutting algorithm
        
    Returns:
        Annotated image
    """
    # Convert to PIL Image for easier drawing
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # Create a separate overlay for the masks
    overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Calculate line width based on image width
    line_width = calculate_line_width(image.shape[1])  # image.shape[1] is the width
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
    
    # Generate colors for units and compartments
    unit_colors = generate_colors(len(storage_units))
    
    # Draw storage units
    for i, unit in enumerate(storage_units):
        unit_color = unit_colors[i]
        
        # Draw unit segmentation mask if available
        if unit.mask is not None and isinstance(unit.mask, np.ndarray):
            try:
                # Convert mask to binary image if it's not already
                if unit.mask.dtype != bool:
                    mask_binary = unit.mask > 0
                else:
                    mask_binary = unit.mask
                
                # Find contours of the mask
                mask_uint8 = mask_binary.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw the contours on the overlay
                for contour in contours:
                    # Smooth the contour with enhanced smoothing
                    smoothed_contour = smooth_contour(contour, epsilon_factor=epsilon_factor, apply_chaikin=True, chaikin_iterations=chaikin_iterations)
                    
                    # Convert contour points to a list of tuples for PIL
                    contour_points = [tuple(point[0]) for point in smoothed_contour]
                    
                    # Fill the contour with semi-transparent color
                    if len(contour_points) > 2:  # Need at least 3 points to draw a polygon
                        overlay_draw.polygon(contour_points, fill=(*unit_color, int(255 * alpha)))
                        
                        # Create darker color for the border
                        border_color = darken_color(unit_color)
                        
                        # Draw the contour outline with relative width and darker color
                        overlay_draw.line(contour_points + [contour_points[0]], fill=(*border_color, 255), width=line_width)
                
                # Find a good position for the label
                if show_labels and len(contours) > 0:
                    # Use the largest contour for label placement
                    largest_contour = max(contours, key=cv2.contourArea)
                    moments = cv2.moments(largest_contour)
                    
                    if moments["m00"] != 0:
                        # Calculate centroid of the contour
                        cx = int(moments["m10"] / moments["m00"])
                        cy = int(moments["m01"] / moments["m00"])
                    else:
                        # Fallback to bounding box center
                        cx, cy = unit.center
                    
                    # Create label
                    label = f"{unit.class_name}"
                    
                    # Get text size
                    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
                    text_width = right - left
                    text_height = bottom - top
                    
                    # Draw label background
                    overlay_draw.rectangle(
                        [(cx - text_width//2 - 2, cy - text_height//2 - 2), 
                         (cx + text_width//2 + 2, cy + text_height//2 + 2)],
                        fill=(*unit_color, 200)
                    )
                    
                    # Draw label text
                    overlay_draw.text(
                        (cx - text_width//2, cy - text_height//2),
                        label,
                        fill=(255, 255, 255, 255),
                        font=font
                    )
            except Exception as e:
                print(f"Warning: Could not apply mask for unit {i}: {e}")
                # Fallback to bounding box if mask processing fails
                draw.rectangle(
                    [(unit.x1, unit.y1), (unit.x2, unit.y2)],
                    outline=unit_color,
                    width=line_width
                )
        else:
            # Fallback to bounding box if no mask is available
            draw.rectangle(
                [(unit.x1, unit.y1), (unit.x2, unit.y2)],
                outline=unit_color,
                width=line_width
            )
            
            # Draw unit label for bounding box
            if show_labels:
                label = f"{unit.class_name}"
                left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
                text_width = right - left
                text_height = bottom - top
                
                draw.rectangle(
                    [(unit.x1, unit.y1 - text_height - 4), (unit.x1 + text_width + 4, unit.y1)],
                    fill=unit_color
                )
                draw.text(
                    (unit.x1 + 2, unit.y1 - text_height - 2),
                    label,
                    fill=(255, 255, 255),
                    font=font
                )
        
        # Generate colors for compartments
        compartment_colors = generate_colors(len(unit.compartments))
        
        # Draw compartments
        for j, compartment in enumerate(unit.compartments):
            comp_color = compartment_colors[j]
            
            # Draw compartment segmentation mask if available
            if compartment.mask is not None and isinstance(compartment.mask, np.ndarray):
                try:
                    # Convert mask to binary image if it's not already
                    if compartment.mask.dtype != bool:
                        mask_binary = compartment.mask > 0
                    else:
                        mask_binary = compartment.mask
                    
                    # Find contours of the mask
                    mask_uint8 = mask_binary.astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Draw the contours on the overlay
                    for contour in contours:
                        # Smooth the contour with enhanced smoothing
                        smoothed_contour = smooth_contour(contour, epsilon_factor=epsilon_factor, apply_chaikin=True, chaikin_iterations=chaikin_iterations)
                        
                        # Convert contour points to a list of tuples for PIL
                        contour_points = [tuple(point[0]) for point in smoothed_contour]
                        
                        # Fill the contour with semi-transparent color
                        if len(contour_points) > 2:  # Need at least 3 points to draw a polygon
                            overlay_draw.polygon(contour_points, fill=(*comp_color, int(255 * alpha)))
                            
                            # Create darker color for the border
                            border_color = darken_color(comp_color)
                            
                            # Draw the contour outline with relative width and darker color
                            overlay_draw.line(contour_points + [contour_points[0]], fill=(*border_color, 255), width=line_width)
                    
                    # Find a good position for the label
                    if show_labels and len(contours) > 0:
                        # Use the largest contour for label placement
                        largest_contour = max(contours, key=cv2.contourArea)
                        moments = cv2.moments(largest_contour)
                        
                        if moments["m00"] != 0:
                            # Calculate centroid of the contour
                            cx = int(moments["m10"] / moments["m00"])
                            cy = int(moments["m01"] / moments["m00"])
                        else:
                            # Fallback to bounding box center
                            cx, cy = compartment.center
                        
                        # Create label
                        label = f"{compartment.class_name}"
                        
                        # Get text size
                        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
                        text_width = right - left
                        text_height = bottom - top
                        
                        # Draw label background
                        overlay_draw.rectangle(
                            [(cx - text_width//2 - 2, cy - text_height//2 - 2), 
                             (cx + text_width//2 + 2, cy + text_height//2 + 2)],
                            fill=(*comp_color, 200)
                        )
                        
                        # Draw label text
                        overlay_draw.text(
                            (cx - text_width//2, cy - text_height//2),
                            label,
                            fill=(255, 255, 255, 255),
                            font=font
                        )
                except Exception as e:
                    print(f"Warning: Could not apply mask for compartment {j} in unit {i}: {e}")
                    # Fallback to bounding box if mask processing fails
                    draw.rectangle(
                        [(compartment.x1, compartment.y1), (compartment.x2, compartment.y2)],
                        outline=comp_color,
                        width=line_width
                    )
            else:
                # Fallback to bounding box if no mask is available
                draw.rectangle(
                    [(compartment.x1, compartment.y1), (compartment.x2, compartment.y2)],
                    outline=comp_color,
                    width=line_width
                )
                
                # Draw compartment label for bounding box
                if show_labels:
                    label = f"{compartment.class_name}"
                    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
                    text_width = right - left
                    text_height = bottom - top
                    
                    draw.rectangle(
                        [(compartment.x1, compartment.y1 - text_height - 4), 
                         (compartment.x1 + text_width + 4, compartment.y1)],
                        fill=comp_color
                    )
                    draw.text(
                        (compartment.x1 + 2, compartment.y1 - text_height - 2),
                        label,
                        fill=(255, 255, 255),
                        font=font
                    )
    
    # Composite the overlay with the original image
    pil_image = Image.alpha_composite(pil_image.convert('RGBA'), overlay).convert('RGB')
    
    # Convert back to numpy array
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def create_hierarchy_tree(storage_units: List[StorageUnit]) -> str:
    """
    Create an HTML representation of the storage hierarchy as a tree.
    
    Args:
        storage_units: List of detected storage units
        
    Returns:
        HTML string representing the hierarchy tree
    """
    html = """
    <style>
    .tree {
        font-family: Arial, sans-serif;
        margin: 20px;
    }
    .tree ul {
        padding-left: 20px;
    }
    .tree li {
        list-style-type: none;
        margin: 10px;
        position: relative;
    }
    .tree li::before {
        content: "";
        position: absolute;
        top: -5px;
        left: -20px;
        border-left: 1px solid #ccc;
        border-bottom: 1px solid #ccc;
        width: 20px;
        height: 15px;
    }
    .tree li:first-child::before {
        top: 10px;
    }
    .tree ul li:last-child::before {
        height: 25px;
    }
    .tree ul li:last-child {
        border-left: none;
    }
    .tree li span {
        display: inline-block;
        padding: 5px 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
        background-color: #f8f9fa;
    }
    .unit span {
        background-color: #e6f7ff;
        border-color: #91d5ff;
    }
    .compartment span {
        background-color: #f6ffed;
        border-color: #b7eb8f;
    }
    </style>
    <div class="tree">
        <ul>
    """
    
    # Add storage units to the tree
    for unit in storage_units:
        unit_info = f"{unit.class_name} ({unit.confidence:.2f}) - {unit.width}x{unit.height}"
        html += f'<li class="unit"><span>{unit_info}</span>'
        
        # Add compartments if any
        if unit.compartments:
            html += '<ul>'
            for comp in unit.compartments:
                comp_info = f"{comp.class_name} ({comp.confidence:.2f}) - {comp.width}x{comp.height}"
                html += f'<li class="compartment"><span>{comp_info}</span></li>'
            html += '</ul>'
        
        html += '</li>'
    
    html += """
        </ul>
    </div>
    """
    
    return html

def crop_image(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop an image to a bounding box.
    
    Args:
        image: Input image
        bbox: Bounding box (x1, y1, x2, y2)
        
    Returns:
        Cropped image
    """
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]

def resize_with_aspect_ratio(
    image: np.ndarray,
    width: int = None,
    height: int = None,
    inter: int = cv2.INTER_AREA
) -> np.ndarray:
    """
    Resize an image while maintaining aspect ratio.
    
    Args:
        image: Input image
        width: Target width
        height: Target height
        inter: Interpolation method
        
    Returns:
        Resized image
    """
    dim = None
    h, w = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    
    return cv2.resize(image, dim, interpolation=inter)

def export_results_to_json(storage_units: List[StorageUnit]) -> Dict[str, Any]:
    """
    Export detection results to a JSON-serializable dictionary.
    
    Args:
        storage_units: List of detected storage units
        
    Returns:
        Dictionary with detection results
    """
    results = []
    
    for unit in storage_units:
        unit_dict = {
            "class": unit.class_name,
            "confidence": float(unit.confidence),
            "bbox": [int(unit.x1), int(unit.y1), int(unit.x2), int(unit.y2)],
            "width": int(unit.width),
            "height": int(unit.height),
            "compartments": []
        }
        
        for comp in unit.compartments:
            comp_dict = {
                "class": comp.class_name,
                "confidence": float(comp.confidence),
                "bbox": [int(comp.x1), int(comp.y1), int(comp.x2), int(comp.y2)],
                "width": int(comp.width),
                "height": int(comp.height)
            }
            unit_dict["compartments"].append(comp_dict)
        
        results.append(unit_dict)
    
    return {"storage_units": results}