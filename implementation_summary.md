# Implementation Summary

## Changes Made

I've successfully implemented the post-processing enhancements to the segmentation visualization in `utils.py`. Here's a summary of the changes:

### 1. Added Helper Functions

Added two new helper functions at the top of the file:

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

### 2. Modified Storage Unit Contour Drawing

Changed the contour drawing code for storage units (around line 83) to:
- Apply smoothing to contours
- Use a darker color for borders
- Increase border width from 3px to 7px

Before:
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

After:
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

### 3. Modified Compartment Contour Drawing

Similarly, changed the contour drawing code for compartments (around line 184) to:
- Apply smoothing to contours
- Use a darker color for borders
- Increase border width from 2px to 7px

Before:
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

After:
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

## Testing the Implementation

To test the changes, you can run the application with various images and observe the segmentation visualization. The key improvements to look for are:

1. **Smoother Contours**: The borders should appear less jagged and more natural.
2. **Thicker Borders**: The borders should be 7px wide, making them more visible.
3. **Darker Border Colors**: The borders should be a darker shade of the segment color, providing better contrast.

### Fine-Tuning Parameters

If needed, you can adjust these parameters to optimize the visualization:

1. **Darkening Factor**: In the `darken_color` function, adjust the `factor` parameter (default: 0.7)
   - Lower values make borders darker
   - Higher values make borders closer to the original color

2. **Smoothing Factor**: In the `smooth_contour` function, adjust the `epsilon_factor` parameter (default: 0.005)
   - Higher values create smoother but less accurate contours
   - Lower values preserve more detail but may remain jagged

### Alternative Approaches

If the current smoothing approach doesn't provide the desired results, consider these alternatives:

1. **Gaussian Blur**: Apply Gaussian blur to the mask before finding contours
   ```python
   # Apply Gaussian blur to smooth the mask edges
   smoothed_mask = cv2.GaussianBlur(mask_uint8, (5, 5), 0)
   contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   ```

2. **Chaikin's Corner Cutting**: Implement a more advanced smoothing algorithm
   ```python
   def chaikin_corner_cutting(points, iterations=2):
       """Apply Chaikin's corner cutting algorithm for smooth curves."""
       for _ in range(iterations):
           new_points = []
           for i in range(len(points) - 1):
               p0 = points[i]
               p1 = points[i + 1]
               q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
               r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
               new_points.extend([q, r])
           points = new_points
       return np.array(points).reshape(-1, 1, 2).astype(np.int32)