# Resize and Pad Image Dataset - Pascal VOC format
To resize image dataset with padding method (black bars either on the sides or top and bottom) to ensure the aspect ratio of each original image is remained. This script is created for image dataset with annotations in XML format (pascal VOC format).

The resulting resized + padded image and its rescaled bounding box(es) values are saved.

## Running of Script
Run main python file in terminal - if resized image height and width is 640 * 640:

`python resize_pad_img.py --ann_path ./path/to/annotations --img_path ./path/to/imagedataset --resized_ap ./path/to/resized_annotation --resized_ip ./path/to/resized_images -oh 640 -ow 640`