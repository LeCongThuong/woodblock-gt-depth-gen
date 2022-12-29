import cv2
import math
from parsing import get_border_points_from_via_file, read_json_file, write_json_file
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def rotate_imgs(image, border_points):
    horizontal_vect = border_points[1] - border_points[0]
    rotated_angle = math.degrees(math.atan2(horizontal_vect[1], horizontal_vect[0]))
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=rotated_angle, scale=1)
    # rotate the image using cv2.warpAffine
    rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
    return rotated_image, rotate_matrix


def rotate_bboxes_from_anno_file(sino_nom_file, rotated_matrix):
    anno_info = read_json_file(sino_nom_file)
    bboxes_info = anno_info["bboxes"]
    new_bboxes_list = []
    for bbox in bboxes_info:
        x_min, y_min, x_max, y_max = bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]
        t_l = np.asarray([x_min, y_min, 1]).reshape((-1, 1))
        r_b = np.asarray([x_max, y_max, 1]).reshape((-1, 1))
        new_t_l = rotated_matrix@t_l
        new_r_b = rotated_matrix@r_b
        bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"] = new_t_l[0][0], new_t_l[1][0], new_r_b[0][0], new_r_b[1][0]
        new_bboxes_list.append(bbox)
    anno_info["bboxes"] = new_bboxes_list
    return anno_info


if __name__ == '__main__':
    horizon_path_list = list(Path("data/horizontal_points").glob("*.json"))
    wb_id_list = ["_".join(horizon_path.stem.split("_")[-2:]) for horizon_path in horizon_path_list]
    for wb_id in wb_id_list:
        border_points = f"data/horizontal_points/nb_horizon_{wb_id}.json"
        img_path = f"data/raw/{wb_id}/nb_{wb_id}.png"
        via_anno_path = f"data/raw/{wb_id}/nb_{wb_id}.json"
        img = cv2.imread(img_path)
        border_points = get_border_points_from_via_file(border_points, mirror=True, key_src="nb")
        rotated_img, rotate_matrix = rotate_imgs(img, border_points)
        cv2.imwrite(f"data/raw/{wb_id}/nb_aligned_{wb_id}.png", rotated_img)
        anno_info = rotate_bboxes_from_anno_file(via_anno_path, rotate_matrix)
        write_json_file(anno_info, f"data/raw/{wb_id}/nb_aligned_{wb_id}.json")

