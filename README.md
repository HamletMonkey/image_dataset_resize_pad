# Resize and Pad Image Dataset - Pascal VOC format
To resize image dataset with padding method (black bars either on the sides or top and bottom) to ensure the aspect ratio of each original image is remained. This script is created for image dataset with annotations in XML format (pascal VOC format).

The resulting resized + padded image and its rescaled bounding box(es) values are saved.

## Running of Script
Run python file in terminal - if resizing image to a width and height of 640 * 640:

`python resize_pad_img.py -ow 640 -oh 640 --imgpath ./path/to/images --annpath ./path/to/annotations`