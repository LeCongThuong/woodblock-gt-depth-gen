import open3d as o3d
import numpy as np
from model_transform_utils import get_surface_equation_coeffs, get_3d_transform_matrix, transform_points, read_stl_file
from pathlib import Path


def raw_pitch_transform_3d_points(pc_woodblock_points, pc_floor_points, np_border_points, mirror=False):
    surface_coeffs = get_surface_equation_coeffs(np.asarray(pc_floor_points.vertices), order=1)
    transform_matrix = get_3d_transform_matrix(surface_coeffs, np_border_points)
    pc_woodblock_points.vertices = o3d.utility.Vector3dVector(transform_points(np.asarray(pc_woodblock_points.vertices),
                                                                               transform_matrix,
                                                                               mirror))
    pc_woodblock_points = o3d.geometry.TriangleMesh.compute_triangle_normals(pc_woodblock_points)
    pc_woodblock_points.remove_duplicated_vertices()
    return pc_woodblock_points


def pitch_transform_3d_points(pc_floor_points, pc_woodblock_points, mirror=False):
    surface_coeffs = get_surface_equation_coeffs(np.asarray(pc_floor_points.vertices), order=1)
    transform_matrix = get_3d_transform_matrix(surface_coeffs)
    pc_woodblock_points.vertices = o3d.utility.Vector3dVector(transform_points(np.asarray(pc_woodblock_points.vertices),
                                                                               transform_matrix,
                                                                               mirror))
    pc_woodblock_points = o3d.geometry.TriangleMesh.compute_triangle_normals(pc_woodblock_points)
    pc_woodblock_points.remove_duplicated_vertices()
    return pc_woodblock_points


if __name__ == '__main__':
    woodblock_file_path = "/media/hmi/Expansion/MOCBAN_TEST_OK/whole_woodblock/07888_woodblock_surface_2.stl"
    surface_file_path = "/media/hmi/Expansion/MOCBAN_TEST_OK/floor_woodblock_points/07888_woodblock_floor_2.stl"
    saved_stl_path = "notebooks/woodblock_z/"
    pc_surface_points = read_stl_file(surface_file_path)
    pc_woodblock_points = read_stl_file(woodblock_file_path)
    pc_woodblock_points = pitch_transform_3d_points(pc_surface_points, pc_woodblock_points)
    o3d.io.write_triangle_mesh(f'{saved_stl_path}/07888_woodblock_surface_1_z.stl', pc_woodblock_points)
