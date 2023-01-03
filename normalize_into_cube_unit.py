import open3d as o3d
import numpy as np
from copy import deepcopy
from pathlib import Path
import os


def normalize_into_cube_unit(stl_file):
    obj = o3d.io.read_triangle_mesh(stl_file)
    bbox_max = obj.get_max_bound()
    bbox_min = obj.get_min_bound()
    scale_ratio = 1 / max(bbox_max - bbox_min)
    scaled_test_obj = obj.scale(scale_ratio, center=np.zeros((3, 1)))
    scaled_bbox_min = scaled_test_obj.get_min_bound()
    scaled_bbox_max = scaled_test_obj.get_max_bound()
    translated_scaled_test_obj = scaled_test_obj.translate((scaled_bbox_max + scaled_bbox_min) / 2)
    normalized_obj = o3d.geometry.TriangleMesh.compute_triangle_normals(translated_scaled_test_obj)
    normalized_obj.remove_duplicated_vertices()
    return normalized_obj


def scale_to_z(a, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-z_scale
        Optionally specify the data type of the output (default is uint16)
    """
    bg_mask = np.full(a.shape, 1)  # 255)
    img = deepcopy(a)  # (((a - min_depth) / float(max_depth - min_depth)) * 254)

    depth_img = np.where(a == np.inf, bg_mask, img)
    return depth_img.astype(dtype)


def ray_tracing_fix_depth_map(pc_mesh, side_range=(-12, 12), fwd_range=(-12, 12), res=(2000, 2000, 255),
                              z_max_camera=10, max_z_distance=16):
    t_pc_mesh = o3d.t.geometry.TriangleMesh.from_legacy(pc_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(t_pc_mesh)
    side_width = side_range[1] - side_range[0]
    fwd_height = fwd_range[1] - fwd_range[0]
    ratio_width = side_width / res[0]
    ratio_height = fwd_height / res[1]
    x_range = side_range[0] + np.arange(res[0]) * ratio_width
    y_range = fwd_range[1] - np.arange(res[1]) * ratio_height
    x_mesh, y_mesh = np.meshgrid(x_range, y_range)
    x_mesh_flat = x_mesh.reshape((-1,))
    y_mesh_flat = y_mesh.reshape((-1,))
    z_mesh_flat = np.full((x_mesh_flat.shape[0],), z_max_camera)
    direction = np.repeat(np.asarray([0, 0, -1]).reshape((1, -1)), z_mesh_flat.shape[0], axis=0)
    point_mesh = np.stack((x_mesh_flat, y_mesh_flat, z_mesh_flat), axis=1)
    rays = np.concatenate((point_mesh, direction), axis=1)
    rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    ans = scene.cast_rays(rays, nthreads=8)
    pixel_values = ans['t_hit'].numpy()
    pixel_values = pixel_values.reshape((res[0], res[1]))

    normalized_pixel_values = scale_to_z(pixel_values, dtype=np.float)
    # img_inverted_matrix = np.array([[ratio_width, 0, 0, side_range[0]], [0, -ratio_height, 0, fwd_range[1]],
    #                                 [0, 0, -max_z_distance / 254.0, z_max_camera], [0, 0, 0, 1]])
    return normalized_pixel_values


def convert_normalized_pc_to_depth_map(stl_path, res=(256, 256, 255)):
    """
    Main function to convert the .stl model into depth map images.
    :param stl_path: 3D triangle_mesh object or the file path string
    :param res: the resolution of output depth map image
    :return:
        + normalized_depth_img: depth image with the input resolution.
        Because the camera position is at z_max + 0.5, so [0, 254] range in images: expresses distance [z_max + 0.5, z_min + 0.5].
        Value 255 expresses the infinity
        + img_inverted_matrix: the matrix will be used when you want to get back almost point cloud model from depth map (can not convert back completely) through matrix multi.
    """
    if isinstance(stl_path, str):
        pc_mesh = o3d.io.read_triangle_mesh(stl_path)
    else:
        pc_mesh = stl_path
    normalized_depth_img = ray_tracing_fix_depth_map(pc_mesh, side_range=(-0.5, 0.5), fwd_range=(-0.5, 0.5), res=res,
                                                  z_max_camera=0.5, max_z_distance=1)
    return normalized_depth_img


def run_a_file(file_path, stl_dest_path, depth_dest_path):
    normalized_obj = normalize_into_cube_unit(file_path)
    o3d.io.write_triangle_mesh(stl_dest_path, normalized_obj)
    depth_img = convert_normalized_pc_to_depth_map(normalized_obj)
    with open(depth_dest_path, "wb") as f:
        np.save(f, depth_img)


def run_files(parent_dir):
    wb_id_list = [wb_path.name for wb_path in list(Path(parent_dir).glob("*"))]
    for wb_id in wb_id_list:
        wb_id_character_dir = os.path.join(parent_dir, wb_id, "character_xyz")
        stl_character_dir = os.path.join(wb_id_character_dir, "stl")

        normalized_stl_path = os.path.join(wb_id_character_dir, "normalized_stl")
        Path(normalized_stl_path).mkdir(exist_ok=True, parents=True)
        normalized_depth_path = os.path.join(wb_id_character_dir, "normalized_depth")
        Path(normalized_depth_path).mkdir(exist_ok=True, parents=True)
        stl_character_path_list = list(Path(stl_character_dir).glob("*.stl"))
        for stl_character_path in stl_character_path_list:
            name_index = stl_character_path.stem
            dest_normalized_stl_path = os.path.join(normalized_stl_path, f"{name_index}.stl")
            dest_normalized_depth_path = os.path.join(normalized_depth_path, f"{name_index}.npy")
            run_a_file(str(stl_character_path), str(dest_normalized_stl_path), str(dest_normalized_depth_path))


if __name__ == '__main__':
    parent_dir = "data/output"
    run_files(parent_dir)

