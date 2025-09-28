from aug import *
import os
import cv2
import copy
import random
import numpy as np
import time
import json
from PIL import Image, ImageDraw

def get_result(file_name):
    file_inp = f"{file_name}.jpg"
    file_out = f"out_{file_name}.jpg"

    input_A = cv2.imread(file_inp)
    input_B = copy.deepcopy(input_A)
    beta_contrast = get_beta_contrast_by_histogram(input_B)

    angle_random = random.choice([-20, -15, -10, 10, 15, 20])

    start_time = time.time()
    input_B = rotate_image(input_B, angle=angle_random)
    end_time = time.time()
    print(f"Rotation time: {end_time - start_time:.4f} seconds")

    #input_B = adjust_contrast(input_B, alpha=1.5, beta=beta_contrast)

    start_time = time.time()
    output_aug = frequent_augmentation(input_A, input_B, beta=0.7)
    end_time = time.time()
    print(f"Augmentation time: {end_time - start_time:.4f} seconds")

    output_final = combine_images_with_labels(input_A, input_B, output_aug, ref_img_title=f"Rotate {angle_random} - Beta: {beta_contrast}",)
    cv2.imwrite(file_out, output_final)
    print(f"Output saved as {file_out}")

def json_array_to_image_mask(json_data_array, image_size=None):
    """
    Convert an array of JSON segmentation data to a single binary image mask using OpenCV.

    Args:
        json_data_array (list): List of JSON data, each containing 'segmentation' (list of [x, y] coordinates).
        image_size (tuple): Optional (width, height) of the output mask. If None, derived from all segmentation points.

    Returns:
        numpy.ndarray: Binary image mask (0 for background, 255 for segmented regions).
    """
    # Initialize variables for dynamic image size calculation
    all_points = []

    # Collect all segmentation points to determine image size
    for json_data in json_data_array:
        segmentation = json_data['segmentation'][0]  # Assuming single polygon per JSON
        points = np.array([(segmentation[i], segmentation[i + 1]) for i in range(0, len(segmentation), 2)], dtype=np.int32)
        all_points.append(points)

    # Determine image size
    if image_size is None:
        # Calculate min and max coordinates across all polygons
        all_points_flat = np.concatenate(all_points, axis=0)
        x_min, y_min = np.min(all_points_flat, axis=0)
        x_max, y_max = np.max(all_points_flat, axis=0)
        image_width = int(x_max - x_min + 10)  # Add padding
        image_height = int(y_max - y_min + 10)
    else:
        image_width, image_height = image_size

    # Create a blank image
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # Draw each polygon
    for points in all_points:
        cv2.fillPoly(mask, [points], color=255)  # Fill polygon with white (255)

    return mask

def find_image_id(arr_image, file_name):
    """
    Find the index of the image with the specified ID in the array.

    Args:
        arr_image (list): List of image dictionaries.
        image_id (int): The ID to search for.

    Returns:
        int: Index of the image with the specified ID, or -1 if not found.
    """
    for x in arr_image:
        if x["file_name"] == file_name:
            return x
    return None

def get_arr_seg(arr_segmentation, image_id):
    """
    Get the segmentation array for a specific image ID.

    Args:
        arr_segmentation (list): List of segmentation dictionaries.
        image_id (int): The ID of the image to search for.

    Returns:
        list: List of segmentation dictionaries for the specified image ID.
    """
    arr_seg = []
    for x in arr_segmentation:
        if x["image_id"] == image_id:
            arr_seg.append(x)
    return arr_seg

if __name__ == "__main__":
    path_dataset = "/mnt/mmlab2024nas/danh/few-shot-frequency-augment/datasets/camo/images"
    path_anotate = "/mnt/mmlab2024nas/danh/few-shot-frequency-augment/datasets/camo/Annotations/camosplit6_noset/camo_clean_extra_split_1.json"
    json_anotate = json.load(open(path_anotate, "r"))
    arr_mask = [x for x in json_anotate["images"]]
    arr_segmentation = [x for x in json_anotate["annotations"]]
    file_names = os.listdir(path_dataset)
    print("Length of file_names:", len(file_names))

    count_ = 0
    for file_name in file_names:
        if "_v" not in file_name:
            file_name_without_ext, file_name_ext = file_name.split(".")
            file_name_save_1 = f"{file_name_without_ext}_v1.{file_name_ext}"
            file_name_save_2 = f"{file_name_without_ext}_v2.{file_name_ext}"
            file_name_save_3 = f"{file_name_without_ext}_v3.{file_name_ext}"
            file_name_save_mask = f"{file_name_without_ext}_vmask.{file_name_ext}"
            file_name_save_4 = f"{file_name_without_ext}_v4.{file_name_ext}"
            file_name_save_5 = f"{file_name_without_ext}_v5.{file_name_ext}"

            angle_random = random.choice([-20, -15, -10, 10, 15, 20])

            print(f"Process: {file_name}")
            input_A = cv2.imread(os.path.join(path_dataset, file_name))

            # v1
            path_save = os.path.join(path_dataset, file_name_save_1)
            angle_random = random.choice([-20, -15, -10, 10, 15, 20])
            input_B = copy.deepcopy(input_A)
            input_B = rotate_image(input_B, angle=angle_random)
            output_aug = frequent_augmentation(input_A, input_B, beta=0.7)
            cv2.imwrite(path_save, output_aug)
            print(f"-> File saved: {file_name_save_1}")


            # v2
            path_save = os.path.join(path_dataset, file_name_save_2)
            angle_random = random.choice([-20, -15, -10, 10, 15, 20])
            input_B = copy.deepcopy(input_A)
            input_B = rotate_image(input_B, angle=angle_random)
            input_B = adjust_contrast(input_B, alpha=1.5, beta=3)
            output_aug = frequent_augmentation(input_A, input_B, beta=0.7)
            cv2.imwrite(path_save, output_aug)
            print(f"-> File saved: {file_name_save_2}")


            # v3
            path_save = os.path.join(path_dataset, file_name_save_3)
            angle_random = random.choice([-20, -15, -10, 10, 15, 20])
            input_B = copy.deepcopy(input_A)
            input_B = rotate_image(input_B, angle=angle_random)
            input_B = adjust_contrast(input_B, alpha=2, beta=5)
            output_aug = frequent_augmentation(input_A, input_B, beta=0.7)
            cv2.imwrite(path_save, output_aug)
            print(f"-> File saved: {file_name_save_3}")


            # # v4 v5
            path_save_mask = os.path.join(path_dataset, file_name_save_mask)
            angle_random = random.choice([-20, -15, -10, 10, 15, 20])
            input_B = copy.deepcopy(input_A)
            input_B = rotate_image(input_B, angle=angle_random)
            input_B = adjust_contrast(input_B, alpha=2, beta=5)
            output_aug = frequent_augmentation(input_A, input_B, beta=0.7)


            try:
                image_obj = find_image_id(arr_mask, file_name)
                arr_seg = get_arr_seg(arr_segmentation, image_obj["id"])
                mask = json_array_to_image_mask(arr_seg, image_size=(input_A.shape[1], input_A.shape[0]))
                cv2.imwrite(path_save_mask, mask)
                print(f"-> File saved: {file_name_save_mask}")

                if len(input_A.shape) == 3:
                    mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                else:
                    mask_3c = mask

                path_save = os.path.join(path_dataset, file_name_save_4)
                output_aug_v4 = input_A.copy()
                output_aug_v4[mask_3c == 255] = output_aug[mask_3c == 255]
                cv2.imwrite(path_save, output_aug_v4)
                print(f"-> File saved: {file_name_save_4}")

                path_save = os.path.join(path_dataset, file_name_save_5)
                output_aug_v5 = input_A.copy()
                output_aug_v5[mask_3c != 255] = output_aug[mask_3c != 255]
                cv2.imwrite(path_save, output_aug_v5)

                print(f"-> File saved: {file_name_save_5}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")


            count_ = count_ + 1
            print(f"Count: {count_}")

