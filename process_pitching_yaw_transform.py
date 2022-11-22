from pointcloud_to_depth import convert_pc_to_depth_map
from model_transform import raw_pitch_transform_3d_points
from parsing import get_point_from_via_file
import argparse
from pathlib import Path
from model_transform_utils import read_stl_file, get_border_points_from_triangle_mesh
import cv2
from tqdm.auto import tqdm
import gc
import open3d as o3d
import numpy as np
from parsing import get_border_points_from_via_file


def process_pitching_yaw_transform(woodblock_path, floor_path, surface_path, border_path, matrix_z_path, mirror,
                                   dest_surface_path, dest_matrix_xyz_path, dest_depth_xyz_path,
                                   dest_woodblock_xyz_path):
        try:
            pc_floor_points = read_stl_file(floor_path)
            pc_surface_points = read_stl_file(surface_path)
            pc_woodblock_points = read_stl_file(woodblock_path)
            matrix_z = np.load(str(matrix_z_path))
            border_points = get_border_points_from_via_file(border_path, mirror)
            border_points = np.hstack((border_points, np.zeros((border_points.shape[0], 1))))
            pc_woodblock_points, pc_surface_points = raw_pitch_transform_3d_points(pc_woodblock_points,
                                                                                   pc_surface_points,
                                                                                   pc_floor_points,
                                                                                   border_points,
                                                                                   matrix_z,
                                                                                   mirror)
            normalized_inverted_matrix, normalized_depth_img = convert_pc_to_depth_map(pc_woodblock_points)

            cv2.imwrite(str(dest_depth_xyz_path), normalized_depth_img)
            o3d.io.write_triangle_mesh(str(dest_woodblock_xyz_path), pc_woodblock_points)
            o3d.io.write_triangle_mesh(str(dest_surface_path), pc_surface_points)
            with open(str(dest_matrix_xyz_path), 'wb') as f:
                np.save(f, normalized_inverted_matrix)

            del pc_woodblock_points
            del normalized_depth_img
            del pc_surface_points
            gc.collect()

        except Exception as e:
            print(e)
            print(f"Error at {Path(woodblock_path).stem} in process_pitching_yaw_transform function")
            raise Exception()




