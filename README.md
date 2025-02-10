# Dominant Color Extraction for Image Segmentation

## Overview
This project provides a method to extract dominant colors from an image using K-Means clustering. Extracted colors can be used in various applications such as image segmentation, object recognition, color-based filtering, and visual analysis. The implementation efficiently processes images to determine the most prominent color groups, segmenting images based on these dominant colors.

## Features
- Reads an image in BGR format and converts it to RGB for proper visualization.
- Applies K-Means clustering to extract dominant colors.
- Displays the extracted colors as color swatches for easy identification.
- Segments the original image by replacing pixel values with their corresponding dominant colors.
- Provides a visual comparison between the segmented image and the original image.
- Allows customization of the number of dominant colors extracted.

## Installation
To run this project, follow these steps:

```bash
# Clone the repository
git clone https://github.com/Akshatbhatnagar908/Dominant-Color-Extraction-for-Image-Segmentation.git

# Navigate to the project directory
cd Dominant-Color-Extraction-for-Image-Segmentation

# Install required dependencies
pip install -r requirements.txt
```

## Usage
To extract dominant colors and segment an image, use the following Python script:

```python
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans

# Load and convert the image to RGB
im = cv2.imread('image.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
original_image = im

# Flatten image pixels into a 2D array
all_pixels = im.reshape((-1, 3))

# Apply K-Means clustering to identify dominant colors
num_clusters = 4  # Adjust this for more or fewer dominant colors
km = KMeans(n_clusters=num_clusters)
km.fit(all_pixels)
centers = np.array(km.cluster_centers_, dtype='uint8')

# Display extracted dominant colors
plt.figure(figsize=(6,2))
for i, each_col in enumerate(centers, 1):
    plt.subplot(1, num_clusters, i)
    plt.axis('off')
    color_patch = np.zeros((100, 100, 3), dtype='uint8')
    color_patch[:, :, :] = each_col
    plt.imshow(color_patch)
plt.show()

# Segment the image using dominant colors
new_img = np.zeros((all_pixels.shape[0], 3), dtype='uint8')
for i in range(new_img.shape[0]):
    new_img[i] = centers[km.labels_[i]]
new_img = new_img.reshape(original_image.shape)

# Display the segmented image
plt.imshow(new_img)
plt.title("Segmented Image")
plt.show()

# Compare segmented and original images side by side
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Segmented Image")
plt.imshow(new_img)
plt.subplot(1,2,2)
plt.title("Original Image")
plt.imshow(original_image)
plt.show()
```

## Parameters
- `num_clusters`: Specifies the number of dominant colors to extract. Higher values result in finer segmentation.

## Dependencies
Ensure you have the following dependencies installed:
- Python 3.x
- OpenCV (cv2)
- NumPy
- Scikit-learn
- Matplotlib

## Output
The script provides:
- A visualization of the extracted dominant colors.
- A segmented version of the input image where colors are replaced with their closest dominant color.
- A side-by-side comparison of the segmented image and the original image.

## Applications
- **Image Segmentation**: Used in computer vision for object recognition and scene understanding.
- **Color-Based Filtering**: Useful in image editing and design applications.
- **Visual Analytics**: Helps in color trend analysis and palette generation.

## Contributing
Contributions are welcome! If you find a bug or have a feature request, feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License.

## Author
[AKSHAT BHATNAGAR](https://github.com/Akshatbhatnagar908)
