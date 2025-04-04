This is a report on possible libraries to use in future PoC

# Python Image Segmentation Libraries for Furniture
Below is a comparison of actively maintained Python libraries that support furniture segmentation and were updated in 2024, ordered by popularity:

| Library | Initial Release | Last Updated | Furniture Segmentation Capabilities | Popularity Metrics |
|---------|----------------|--------------|-------------------------------------|-------------------|
| Segment Anything Model (SAM) | April 2023 | 2024 | Foundation model capable of segmenting almost any object including all types of furniture | 40K+ GitHub stars, 2M+ downloads, rapid adoption since release |
| Mask R-CNN (Detectron2) | 2017 (paper), 2019 (Detectron2) | 2024 | Trained on COCO dataset which includes furniture classes (chair, couch, bed, etc.) | 25K+ GitHub stars, widely used in research and industry |
| DeepLabV3+ | 2018 | 2024 | Pre-trained on ADE20K and PASCAL VOC datasets which include furniture categories | Millions of downloads, widely adopted in academia and industry |
| MMSegmentation | 2020 | 2024 | Supports models trained on ADE20K which includes furniture classes | 7K+ GitHub stars, growing adoption especially in Asia |
| PyTorch Segmentation Models | ~2019 | 2024 | Models pre-trained on ADE20K dataset which includes furniture categories | 6K+ GitHub stars, popular in PyTorch ecosystem |

## Notes:
- All libraries listed support GPU acceleration for faster inference
- Segment Anything Model (SAM) offers the most flexible approach as it can be prompted in various ways
- Mask R-CNN provides instance-level segmentation (can distinguish individual furniture pieces)
- DeepLabV3+ and MMSegmentation excel at semantic segmentation (identifying furniture types)
- Most libraries can be further fine-tuned on custom furniture datasets if needed

## Links:
Python Image Segmentation Libraries for Furniture (2024)
Here's a curated list of actively maintained image segmentation libraries that support furniture segmentation, ordered by popularity:

### Segment Anything Model (SAM)

Library: segment-anything
Documentation: SAM Documentation
Popularity: 40K+ GitHub stars, 2M+ downloads
Furniture Support: Foundation model capable of segmenting virtually any object including furniture


### Mask R-CNN (Detectron2)

Library: detectron2
Documentation: Detectron2 Documentation
Popularity: 25K+ GitHub stars
Furniture Support: Pre-trained on COCO dataset with furniture classes (chair, couch, bed, etc.)


### DeepLabV3+

Library: tensorflow/models
Documentation: DeepLab Documentation
Popularity: Millions of downloads
Furniture Support: Models pre-trained on ADE20K and PASCAL VOC include furniture categories


MMSegmentation

Library: mmsegmentation
Documentation: MMSegmentation Documentation
Popularity: 7K+ GitHub stars
Furniture Support: Supports models trained on ADE20K with furniture classes


PyTorch Segmentation Models

Library: segmentation_models.pytorch
Documentation: SMP Documentation
Popularity: 6K+ GitHub stars
Furniture Support: Compatible with pre-trained encoders on ADE20K dataset including furniture categories