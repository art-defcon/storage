# Implementation Code Changes

This document contains the exact code changes needed to implement the segmentation visualization enhancements. Since Architect mode can only edit Markdown files, these changes will need to be implemented by switching to Code mode.

## Changes to `utils.py`

### 1. Add New Helper Functions

Add these new helper functions after the existing imports and helper functions:

```python
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
```

### 2. Replace the Existing `smooth_contour` Function

Replace the current `smooth_contour` function with this enhanced version:

```python
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
```

### 3. Modify the `visualize_segmentation` Function

Make the following changes to the `visualize_segmentation` function:

#### A. Add Line Width Calculation

After the line where the overlay is created (around line 82), add:

```python
# Calculate line width based on image width
line_width = calculate_line_width(image.shape[1])  # image.shape[1] is the width
```

#### B. Update Storage Unit Contour Drawing

Find the section where storage unit contours are drawn (around line 112) and replace it with:

```python
# Draw the contours on the overlay
for contour in contours:
    # Smooth the contour with enhanced smoothing
    smoothed_contour = smooth_contour(contour, epsilon_factor=0.02, apply_chaikin=True, chaikin_iterations=3)
    
    # Convert contour points to a list of tuples for PIL
    contour_points = [tuple(point[0]) for point in smoothed_contour]
    
    # Fill the contour with semi-transparent color
    if len(contour_points) > 2:  # Need at least 3 points to draw a polygon
        overlay_draw.polygon(contour_points, fill=(*unit_color, int(255 * alpha)))
        
        # Create darker color for the border
        border_color = darken_color(unit_color)
        
        # Draw the contour outline with relative width and darker color
        overlay_draw.line(contour_points + [contour_points[0]], fill=(*border_color, 255), width=line_width)
```

#### C. Update Compartment Contour Drawing

Find the section where compartment contours are drawn (around line 220) and replace it with:

```python
# Draw the contours on the overlay
for contour in contours:
    # Smooth the contour with enhanced smoothing
    smoothed_contour = smooth_contour(contour, epsilon_factor=0.02, apply_chaikin=True, chaikin_iterations=3)
    
    # Convert contour points to a list of tuples for PIL
    contour_points = [tuple(point[0]) for point in smoothed_contour]
    
    # Fill the contour with semi-transparent color
    if len(contour_points) > 2:  # Need at least 3 points to draw a polygon
        overlay_draw.polygon(contour_points, fill=(*comp_color, int(255 * alpha)))
        
        # Create darker color for the border
        border_color = darken_color(comp_color)
        
        # Draw the contour outline with relative width and darker color
        overlay_draw.line(contour_points + [contour_points[0]], fill=(*border_color, 255), width=line_width)
```

#### D. Update Fallback Bounding Box Drawing

For the fallback bounding box drawing (when mask processing fails or no mask is available), update the width parameter to use the calculated line width:

For storage units (around line 168 and 177):
```python
draw.rectangle(
    [(unit.x1, unit.y1), (unit.x2, unit.y2)],
    outline=unit_color,
    width=line_width
)
```

For compartments (around line 276 and 284):
```python
draw.rectangle(
    [(compartment.x1, compartment.y1), (compartment.x2, compartment.y2)],
    outline=comp_color,
    width=line_width
)
```

## Complete Implementation

The complete implementation involves:

1. Adding the new helper functions
2. Replacing the existing `smooth_contour` function
3. Modifying the `visualize_segmentation` function to:
   - Calculate the relative line width
   - Use enhanced smoothing for contours
   - Apply relative line width to all borders

After these changes, the segmentation visualization will have:
- Border lines with width proportional to the image size (0.3% of width, minimum 1px)
- Smoother contours with fewer vector points
- Rounded corners instead of sharp angles

## Next Steps

1. Switch to Code mode to implement these changes
2. Test the implementation with various images
3. Adjust parameters if needed for optimal results