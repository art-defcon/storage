# Implementation Details for Segmentation Post-Processing

This document provides detailed implementation instructions for adding smoothed border lines to the segmentation visualization.

## Overview of Changes

We'll be modifying the `utils.py` file to enhance the segmentation visualization with:
1. Thicker border lines (7px)
2. Darker border colors
3. Smoothed contours to reduce jaggedness

## Step-by-Step Implementation

### 1. Add Helper Functions

First, add these helper functions at the top of the `utils.py` file, after the existing imports:

```python
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

def smooth_contour(contour, epsilon_factor=0.005):
    """
    Smooth a contour using Douglas-Peucker algorithm.
    
    Args:
        contour: OpenCV contour
        epsilon_factor: Smoothing factor (higher = more smoothing)
        
    Returns:
        Smoothed contour
    """
    perimeter = cv2.arcLength(contour, True)
    epsilon = epsilon_factor * perimeter
    return cv2.approxPolyDP(contour, epsilon, True)
```

### 2. Modify the `visualize_segmentation` Function

Now, we need to update the `visualize_segmentation` function to use our new helper functions. Here are the specific changes needed:

#### For Storage Units (around line 83)

Find this code block:

```python
# Draw the contours on the overlay
for contour in contours:
    # Convert contour points to a list of tuples for PIL
    contour_points = [tuple(point[0]) for point in contour]
    
    # Fill the contour with semi-transparent color
    if len(contour_points) > 2:  # Need at least 3 points to draw a polygon
        overlay_draw.polygon(contour_points, fill=(*unit_color, int(255 * alpha)))
        # Draw the contour outline
        overlay_draw.line(contour_points + [contour_points[0]], fill=(*unit_color, 255), width=3)
```

Replace it with:

```python
# Draw the contours on the overlay
for contour in contours:
    # Smooth the contour
    smoothed_contour = smooth_contour(contour)
    
    # Convert contour points to a list of tuples for PIL
    contour_points = [tuple(point[0]) for point in smoothed_contour]
    
    # Fill the contour with semi-transparent color
    if len(contour_points) > 2:  # Need at least 3 points to draw a polygon
        overlay_draw.polygon(contour_points, fill=(*unit_color, int(255 * alpha)))
        
        # Create darker color for the border
        border_color = darken_color(unit_color)
        
        # Draw the contour outline with thicker width and darker color
        overlay_draw.line(contour_points + [contour_points[0]], fill=(*border_color, 255), width=7)
```

#### For Compartments (around line 184)

Find this code block:

```python
# Draw the contours on the overlay
for contour in contours:
    # Convert contour points to a list of tuples for PIL
    contour_points = [tuple(point[0]) for point in contour]
    
    # Fill the contour with semi-transparent color
    if len(contour_points) > 2:  # Need at least 3 points to draw a polygon
        overlay_draw.polygon(contour_points, fill=(*comp_color, int(255 * alpha)))
        # Draw the contour outline
        overlay_draw.line(contour_points + [contour_points[0]], fill=(*comp_color, 255), width=2)
```

Replace it with:

```python
# Draw the contours on the overlay
for contour in contours:
    # Smooth the contour
    smoothed_contour = smooth_contour(contour)
    
    # Convert contour points to a list of tuples for PIL
    contour_points = [tuple(point[0]) for point in smoothed_contour]
    
    # Fill the contour with semi-transparent color
    if len(contour_points) > 2:  # Need at least 3 points to draw a polygon
        overlay_draw.polygon(contour_points, fill=(*comp_color, int(255 * alpha)))
        
        # Create darker color for the border
        border_color = darken_color(comp_color)
        
        # Draw the contour outline with thicker width and darker color
        overlay_draw.line(contour_points + [contour_points[0]], fill=(*border_color, 255), width=7)
```

### 3. Alternative Smoothing Approach (Optional)

If the Douglas-Peucker algorithm doesn't provide sufficient smoothing, you can try this alternative approach using Gaussian blur:

Replace the contour finding code:

```python
# Find contours of the mask
mask_uint8 = mask_binary.astype(np.uint8) * 255
contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

With:

```python
# Find contours of the mask with smoothing
mask_uint8 = mask_binary.astype(np.uint8) * 255
# Apply Gaussian blur to smooth the mask edges
smoothed_mask = cv2.GaussianBlur(mask_uint8, (5, 5), 0)
contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

### 4. Fine-Tuning Parameters

You may need to adjust these parameters for optimal results:

1. **Darkening factor**: Adjust the `factor` parameter in `darken_color` (default: 0.7)
   - Lower values make borders darker
   - Higher values make borders closer to the original color

2. **Smoothing factor**: Adjust the `epsilon_factor` parameter in `smooth_contour` (default: 0.005)
   - Higher values create smoother but less accurate contours
   - Lower values preserve more detail but may remain jagged

3. **Gaussian blur kernel size**: If using the alternative approach, adjust the kernel size (default: (5, 5))
   - Larger kernels create more smoothing
   - Smaller kernels preserve more detail

## Complete Modified Function

For reference, here's the complete `visualize_segmentation` function with all changes applied:

```python
def visualize_segmentation(
    image: np.ndarray,
    storage_units: List[StorageUnit],
    alpha: float = 0.3,
    show_labels: bool = True
) -> np.ndarray:
    """
    Visualize detected storage units and compartments on the image using segmentation masks.
    
    Args:
        image: Input image
        storage_units: List of detected storage units
        alpha: Transparency of the overlay
        show_labels: Whether to show labels
        
    Returns:
        Annotated image
    """
    # Convert to PIL Image for easier drawing
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # Create a separate overlay for the masks
    overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
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
                    # Smooth the contour
                    smoothed_contour = smooth_contour(contour)
                    
                    # Convert contour points to a list of tuples for PIL
                    contour_points = [tuple(point[0]) for point in smoothed_contour]
                    
                    # Fill the contour with semi-transparent color
                    if len(contour_points) > 2:  # Need at least 3 points to draw a polygon
                        overlay_draw.polygon(contour_points, fill=(*unit_color, int(255 * alpha)))
                        
                        # Create darker color for the border
                        border_color = darken_color(unit_color)
                        
                        # Draw the contour outline with thicker width and darker color
                        overlay_draw.line(contour_points + [contour_points[0]], fill=(*border_color, 255), width=7)
                
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
                    width=3
                )
        else:
            # Fallback to bounding box if no mask is available
            draw.rectangle(
                [(unit.x1, unit.y1), (unit.x2, unit.y2)],
                outline=unit_color,
                width=3
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
                        # Smooth the contour
                        smoothed_contour = smooth_contour(contour)
                        
                        # Convert contour points to a list of tuples for PIL
                        contour_points = [tuple(point[0]) for point in smoothed_contour]
                        
                        # Fill the contour with semi-transparent color
                        if len(contour_points) > 2:  # Need at least 3 points to draw a polygon
                            overlay_draw.polygon(contour_points, fill=(*comp_color, int(255 * alpha)))
                            
                            # Create darker color for the border
                            border_color = darken_color(comp_color)
                            
                            # Draw the contour outline with thicker width and darker color
                            overlay_draw.line(contour_points + [contour_points[0]], fill=(*border_color, 255), width=7)
                    
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
                        width=2
                    )
            else:
                # Fallback to bounding box if no mask is available
                draw.rectangle(
                    [(compartment.x1, compartment.y1), (compartment.x2, compartment.y2)],
                    outline=comp_color,
                    width=2
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
```

## Testing the Implementation

After implementing these changes, you should test the visualization with various images to ensure:

1. The borders are properly smoothed
2. The 7px width is appropriate for your use case
3. The darker color provides good contrast without being too dark

You may need to adjust the parameters based on your specific requirements and the characteristics of your images.