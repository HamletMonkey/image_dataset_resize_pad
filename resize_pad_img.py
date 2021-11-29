import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pascal_voc_writer import Writer
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageOps
import argparse


def resize_pad_XML(
    ANN_PATH, IMG_PATH, OUTPUT_ANN_PATH, OUTPUT_IMG_PATH, output_width, output_height
):

    """
    To resize and pad images, while saving the result images and annotations

    # Arguments
        ANN_PATH: path, annotation folder path
        IMG_PATH: path, image folder path
        OUTPUT_ANN_PATH: path, path to save resized and padded image annotation
        OUTPUT_IMG_PATH: path, path to save resized and padded image
        output_weight: int, resized width
        output_height: int, resized height


    # Outputs
        resized and padded image annotations saved to OUTPUT_ANN_PATH
        resized and padded images saved to OUTPUT_IMG_PATH
    """

    img_id = [f.parts[-1].split(".")[0] for f in Path(ANN_PATH).iterdir()]

    for item in img_id:
        xml_result = []
        tree = ET.parse(os.path.join(ANN_PATH, f"{item}.xml"))
        root = tree.getroot()
        for object in root.findall("object"):
            for value in object.findall("bndbox"):
                xmin = int(value.find("xmin").text)
                ymin = int(value.find("ymin").text)
                xmax = int(value.find("xmax").text)
                ymax = int(value.find("ymax").text)
                xml_result.append([xmin, ymin, xmax, ymax])
        print(len(xml_result))  # amount of bbox per image
        print(xml_result)

        im = Image.open(os.path.join(IMG_PATH, f"{item}.jpg"))
        im_pad = ImageOps.pad(im, (output_width, output_height), color="black")
        im_pad.save(os.path.join(OUTPUT_IMG_PATH, f"{item}.jpg"))
        w, h = im.size

        im_ar = np.float32(w / h)
        output_im_ar = np.float32(output_width / output_height)

        if im_ar < output_im_ar:
            new_h = int(output_height)
            new_w = int(im_ar * new_h)
            # paddings on sides
            diff = abs(output_width - new_w)
            side_pad = diff // 2
            topbottom_pad = 0
        else:
            new_w = int(output_width)
            new_h = int(new_w / im_ar)
            # paddings on top and bottom
            diff = abs(output_height - new_h)
            topbottom_pad = diff // 2
            side_pad = 0

        print(f"new height: {new_h}, new width = {new_w}")
        print(f"side paddings: {side_pad}")
        print(f"top and bottom paddings: {topbottom_pad}")

        # the shifting of bounding box -- with scaling factor
        height_ratio = np.float32(new_h / h)
        width_ratio = np.float32(new_w / w)

        resized_xml = []
        for i in range(len(xml_result)):
            xmin_new = int(xml_result[i][0] * width_ratio + side_pad)
            ymin_new = int(xml_result[i][1] * height_ratio + topbottom_pad)
            xmax_new = int(xml_result[i][2] * width_ratio + side_pad)
            ymax_new = int(xml_result[i][3] * height_ratio + topbottom_pad)
            resized_xml.append([xmin_new, ymin_new, xmax_new, ymax_new])

        print(resized_xml)

        # re-write resized bounding box value into XML annotations
        for size in root.findall("size"):
            size.find("width").text = str(output_width)
            size.find("height").text = str(output_height)
        for index, object in enumerate(root.findall("object")):
            for value in object.findall("bndbox"):
                value.find("xmin").text = str(resized_xml[index][0])
                value.find("ymin").text = str(resized_xml[index][1])
                value.find("xmax").text = str(resized_xml[index][2])
                value.find("ymax").text = str(resized_xml[index][3])
            tree.write(os.path.join(OUTPUT_ANN_PATH, f"{item}.xml"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ann_path",
        type=str,
        default="./xml_ann",
        required=True,
        help="path to XML annotations folder",
    )
    parser.add_argument(
        "--img_path",
        type=str,
        default="./img",
        required=True,
        help="path to image dataset folder",
    )
    parser.add_argument(
        "--resized_ap",
        type=str,
        default="./resized_xml_ann",
        required=True,
        help="path to save rescaled XML annotations",
    )
    parser.add_argument(
        "--resized_ip",
        type=str,
        default="./resized_img",
        required=True,
        help="path to save resized and padded images",
    )
    parser.add_argument(
        "-ow", type=int, required=True, help="desired resized image output width"
    )
    parser.add_argument(
        "-oh", type=int, required=True, help="desired resized image output height"
    )

    args = parser.parse_args()

    resize_pad_XML(
        ANN_PATH=args.ann_path,
        IMG_PATH=args.img_path,
        OUTPUT_ANN_PATH=args.resized_ap,
        OUTPUT_IMG_PATH=args.resized_ip,
        output_width=args.ow,
        output_height=args.oh,
    )
