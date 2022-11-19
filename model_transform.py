import open3d as o3d
import numpy as np
from model_transform_utils import get_surface_equation_coeffs, get_3d_transform_matrix, transform_points, read_stl_file
from pathlib import Path


def raw_pitch_transform_3d_points(pc_surface_points, pc_woodblock_points, np_border_points):
    surface_coeffs = get_surface_equation_coeffs(np.asarray(pc_surface_points.vertices), order=1)
    print(surface_coeffs)
    transform_matrix = get_3d_transform_matrix(surface_coeffs, np_border_points)
    pc_woodblock_points.vertices = o3d.utility.Vector3dVector(transform_points(np.asarray(pc_woodblock_points.vertices), transform_matrix))
    pc_woodblock_points = o3d.geometry.TriangleMesh.compute_triangle_normals(pc_woodblock_points)
    pc_woodblock_points.remove_duplicated_vertices()
    # o3d.io.write_triangle_mesh(f'{dest_dir_path}/{Path(woodblock_path).stem}_aligned.stl', pc_woodblock_points)
    return pc_woodblock_points


def pitch_transform_3d_points(pc_floor_points, pc_woodblock_points):
    # def _do_upside(pc_points):
    #     np_points = np.asarray(pc_points.vertices)
    #     pc_points.vertices = o3d.utility.Vector3dVector(np_points)
    #     pc_points = o3d.geometry.TriangleMesh.compute_triangle_normals(pc_points)
    #     pc_points.remove_duplicated_vertices()
    #     return pc_points

    surface_coeffs = get_surface_equation_coeffs(np.asarray(pc_floor_points.vertices), order=1)
    transform_matrix = get_3d_transform_matrix(surface_coeffs)
    pc_woodblock_points.vertices = o3d.utility.Vector3dVector(transform_points(np.asarray(pc_woodblock_points.vertices), transform_matrix))
    pc_woodblock_points = o3d.geometry.TriangleMesh.compute_triangle_normals(pc_woodblock_points)
    pc_woodblock_points.remove_duplicated_vertices()
    # if upside:
    #     pc_floor_points = _do_upside(pc_floor_points)
    #     pc_surface_points = _do_upside(pc_surface_points)
    return pc_woodblock_points


# need to map from notebook to script file.
if __name__ == '__main__':
    woodblock_file_path = "/media/hmi/Expansion/MOCBAN_TEST_OK/whole_woodblock/07888_woodblock_surface_2.stl"
    surface_file_path = "/media/hmi/Expansion/MOCBAN_TEST_OK/floor_woodblock_points/07888_woodblock_floor_2.stl"
    saved_stl_path = "notebooks/woodblock_z/"
    pc_surface_points = read_stl_file(surface_file_path)
    pc_woodblock_points = read_stl_file(woodblock_file_path)
    pc_woodblock_points = pitch_transform_3d_points(pc_surface_points, pc_woodblock_points)
    o3d.io.write_triangle_mesh(f'{saved_stl_path}/07888_woodblock_surface_1_z.stl', pc_woodblock_points)
