from pointcloud_to_depth import convert_pc_to_depth_map
from model_transform import pitch_transform_3d_points
from model_transform_utils import read_stl_file
import cv2
import gc
import numpy as np
from pathlib import Path


def process_pitching_transform(woodblock_path, floor_path, mirror, dest_depth_path, dest_matrix_z_path):
    try:
        pc_floor_points = read_stl_file(floor_path)
        pc_woodblock_points = read_stl_file(woodblock_path)
        pc_woodblock_points = pitch_transform_3d_points(pc_floor_points, pc_woodblock_points, mirror)

        normalized_inverted_matrix, normalized_depth_img = convert_pc_to_depth_map(pc_woodblock_points)

        with open(dest_matrix_z_path, 'wb') as f:
            np.save(f, normalized_inverted_matrix)
        cv2.imwrite(dest_depth_path, normalized_depth_img)

        del pc_woodblock_points
        del normalized_depth_img
        gc.collect()
    except Exception as e:
        print(e)
        print(f"Error at {Path(woodblock_path).stem} process pitching transform")
        raise Exception(f"{e}")



