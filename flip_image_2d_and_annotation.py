import cv2
import numpy as np
import json


def flip_image_horizontal(image):
    """
    Flip image
    """
    image = cv2.flip(image, 1)
    return image


def flip_annotation_horizontal(np_bbox, image_width):
    """
    Flip annotation
    """
    np_bbox[:, 0] = image_width - 1 - np_bbox[:, 0]
    return np_bbox


def read_json_file(json_path):
    with open(json_path, 'r') as f:
        content = json.load(f)
    return content


def write_json_file(anno, json_path):
    with open(json_path, 'w') as f:
        json.dump(anno, f)


def get_bboxes(via_anno):
    bboxes_points = []
    for key, anno_info in via_anno.items():
        poly_info_list = anno_info["regions"]
        for poly_info in poly_info_list:
            x_point_list = poly_info["shape_attributes"]["all_points_x"]
            y_point_list = poly_info["shape_attributes"]["all_points_y"]
            points = np.stack((x_point_list, y_point_list), axis=-1).astype(np.int16)
            bboxes_points.append(points)
    return bboxes_points


def write_bboxes(via_anno, new_bboxes_points):
    for key, anno_info in via_anno.items():
        poly_info_list = anno_info["regions"]
        for poly_info in poly_info_list:
            poly_info["shape_attributes"]["all_points_x"] = [int(n) for n in new_bboxes_points[0][:, 0]]
            poly_info["shape_attributes"]["all_points_y"] = [int(n) for n in new_bboxes_points[0][:, 1]]
            new_bboxes_points.pop(0)
    return via_anno


def flip_image_and_annotation(image_path, via_json_path, dest_via_json_path, dest_image_path):
    image = cv2.imread(image_path)
    print(image.shape)
    via_anno = read_json_file(via_json_path)
    flipped_img = flip_image_horizontal(image)
    bboxes_points = get_bboxes(via_anno)
    new_bboxes_points = []
    for bbox_points in bboxes_points:
        new_bbox_points = flip_annotation_horizontal(bbox_points, image.shape[1])
        new_bboxes_points.append(new_bbox_points)
    via_anno = write_bboxes(via_anno, new_bboxes_points)
    write_json_file(via_anno, dest_via_json_path)
    cv2.imwrite(dest_image_path, flipped_img)


if __name__ == '__main__':
    image_path = "data/images_2d/08360a_3.jpg"
    via_json_path = "labels/via_labels/08360a_3.json"
    dest_via_json_path = "outputs/via-flip-annotation/flip_08360a_3.json"
    dest_image_path = "outputs/flip-images/flip_08360a_3.jpg"
    flip_image_and_annotation(image_path, via_json_path, dest_via_json_path, dest_image_path)

