import json
import numpy as np
import open3d as o3d


def read_json_file(json_path):
    with open(json_path, 'r') as f:
        content = json.load(f)
    return content


def write_json_file(anno, json_path):
    with open(json_path, 'w') as f:
        json.dump(anno, f)


def get_bboxes_from_via_anno_file(via_json_path, key_src='depth'):
    anno = read_json_file(via_json_path)
    dest_point_list = []
    src_point_list = []
    for anno_key in anno.keys():
        if key_src in anno_key:
            for point_info in anno[anno_key]['regions']:
                dest_point_list.append([point_info['shape_attributes']['cx'], point_info['shape_attributes']['cy']])
        else:
            for point_info in anno[anno_key]['regions']:
                src_point_list.append([point_info['shape_attributes']['cx'], point_info['shape_attributes']['cy']])
    return [src_point_list, dest_point_list]


def get_point_from_via_file(via_json_path, keyword='whole'):
    src_point_list, dest_point_list = get_bboxes_from_via_anno_file(via_json_path, key_src=keyword)
    np_src_point_list = np.asarray(src_point_list)
    np_dest_point_list = np.asarray(dest_point_list)
    return [np_src_point_list, np_dest_point_list]


def parse_bboxes_list_from_sino_nom_anno_file(sino_nom_file):
    anno_info = read_json_file(sino_nom_file)
    bboxes_info = anno_info["bboxes"]
    bboxes_list = []
    for bbox in bboxes_info:
        x_min, y_min, x_max, y_max = bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]
        rect_box_points = np.asarray([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
        bboxes_list.append(rect_box_points)
    return bboxes_list


