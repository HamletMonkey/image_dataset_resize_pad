import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pascal_voc_writer import Writer
import xml.etree.ElementTree as ET
import argparse


def resize_pad_XML(
    ANN_PATH, IMG_PATH, OUTPUT_ANN_PATH, OUTPUT_IMG_PATH, output_height, output_width
):

    """
    To resize and pad images, while saving the result images and annotations

    # Arguments
        ANN_PATH: path, annotation folder path
        IMG_PATH: path, image folder path
        OUTPUT_ANN_PATH: path, path to save resized and padded image annotation
        OUTPUT_IMG_PATH: path, path to save resized and padded image
        output_height: int, resized height
        output_weight: int, resized width

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

        # loading of image
        image = cv2.imread(os.path.join(IMG_PATH, f"{item}.jpg"))
        print(
            f"original image height:{image.shape[0]}, original image width:{image.shape[1]}"
        )

        h, w = image.shape[:2]
        img_ar = w / h  # aspect ratio
        print(img_ar)

        if h > w:
            new_h = output_height
            new_w = int(img_ar * new_h)
            if new_w > output_width:
                new_w = output_width
                new_h = int(new_w / img_ar)
                diff = abs(output_height - new_h)
                if diff % 2 == 0:
                    topbottom_pad = diff // 2
                else:
                    new_h = new_h - 1
                    diff = diff + 1
                    topbottom_pad = diff // 2
                side_pad = 0
            else:
                topbottom_pad = 0
                diff = abs(output_width - new_w)
                if diff % 2 == 0:
                    side_pad = diff // 2
                else:
                    new_w = new_w - 1
                    diff = diff + 1
                    side_pad = diff // 2
        else:
            new_w = output_width
            new_h = int(new_w / img_ar)
            if new_h > output_height:
                new_h = output_height
                new_w = int(img_ar * new_h)
                topbottom_pad = 0
                diff = abs(output_width - new_w)
                if diff % 2 == 0:
                    side_pad = diff // 2
                else:
                    new_w = new_w - 1
                    diff = diff + 1
                    side_pad = diff // 2
            else:
                diff = abs(output_height - new_h)
                if diff % 2 == 0:
                    topbottom_pad = diff // 2
                else:
                    new_h = new_h - 1
                    diff = diff + 1
                    topbottom_pad = diff // 2
                side_pad = 0

        print(f"new height: {new_h}, new width = {new_w}")
        print(f"side paddings: {side_pad}")
        print(f"top and bottom paddings: {topbottom_pad}")

        output = cv2.resize(image, (new_w, new_h))
        output_pad = cv2.copyMakeBorder(
            output,
            topbottom_pad,
            topbottom_pad,
            side_pad,
            side_pad,
            cv2.BORDER_CONSTANT,
            value=0,
        )  # black padding

        cv2.imwrite(os.path.join(OUTPUT_IMG_PATH, f"{item}.jpg"), output_pad)

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
        "--resized_annpath",
        type=str,
        default="./resized_xml_ann",
        required=True,
        help="path to save rescaled XML annotations",
    )
    parser.add_argument(
        "--resized_imgpath",
        type=str,
        default="./resized_img",
        required=True,
        help="path to save resized and padded images",
    )
    parser.add_argument(
        "-ow", type=int, required=True, help="desired resized image output width"
    )
    parser.add_argument(
        "-oh", type=int, required=True, help="desired resized image height"
    )
