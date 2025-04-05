import cv2

def apply_test_time_augmentation(image, use_test_time_augmentation=True):
    """
    Apply test-time augmentation to the image.
    
    Args:
        image (numpy.ndarray): Input image
        use_test_time_augmentation (bool): Whether to use test-time augmentation
        
    Returns:
        list: List of augmented images
    """
    if not use_test_time_augmentation:
        return [image]
    
    augmented_images = [image]  # Original image
    
    # Horizontal flip
    flipped_h = cv2.flip(image, 1)
    augmented_images.append(flipped_h)
    
    # Rotate 90 degrees
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 90, 1.0)
    rotated_90 = cv2.warpAffine(image, rotation_matrix, (w, h))
    augmented_images.append(rotated_90)
    
    # Brightness adjustment
    bright = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
    augmented_images.append(bright)
    
    # Contrast adjustment
    contrast = cv2.convertScaleAbs(image, alpha=1.3, beta=0)
    augmented_images.append(contrast)
    
    return augmented_images