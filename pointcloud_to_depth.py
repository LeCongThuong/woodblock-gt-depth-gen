import numpy as np
import open3d as o3d


def scale_to_z(a, min_depth, max_depth, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-z_scale
        Optionally specify the data type of the output (default is uint16)
    """
    bg_mask = np.full(a.shape, 255)
    img = (((a - min_depth) / float(max_depth - min_depth)) * 254)
    depth_img = np.where(a == np.inf, bg_mask, img)
    return depth_img.astype(dtype)


def get_spatial_limit_of_point_cloud(pc_mesh):
    np_vertices = np.asarray(pc_mesh.vertices)
    x_max = np.max(np_vertices[:, 0])
    x_min = np.min(np_vertices[:, 0])
    y_max = np.max(np_vertices[:, 1])
    y_min = np.min(np_vertices[:, 1])
    z_min = np.min(np_vertices[:, 2])
    z_max = np.max(np_vertices[:, 2])
    spatial_limit = {'x_max': x_max, 'x_min': x_min, 'y_max': y_max, 'y_min': y_min, 'z_min': z_min, 'z_max': z_max}
    return spatial_limit


def ray_tracing_depth_map(pc_mesh, side_range=(-12, 12), fwd_range=(-12, 12), res=(2000, 2000, 255), z_max_camera=10,
                          max_z_distance=16):
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
    _, z_max_depth = -np.sort(-np.unique(pixel_values))[:2]

    z_min_depth = np.min(pixel_values)
    normalized_pixel_values = scale_to_z(pixel_values, z_min_depth, z_max_depth)
    img_inverted_matrix = np.array([[ratio_width, 0, 0, side_range[0]], [0, -ratio_height, 0, fwd_range[1]],
                                    [0, 0, -max_z_distance / 254.0, z_max_camera], [0, 0, 0, 1]])
    return img_inverted_matrix, normalized_pixel_values


def convert_pc_to_depth_map(stl_path, res=(12000, 12000, 255)):
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
    spatial_limit = get_spatial_limit_of_point_cloud(pc_mesh)
    y_length = spatial_limit["y_max"] - spatial_limit["y_min"]
    x_length = spatial_limit["x_max"] - spatial_limit["x_min"]
    # print(spatial_limit)
    if y_length > x_length:
        x_need = (y_length - x_length) / 2
        spatial_limit["x_min"] -= x_need
        spatial_limit["x_max"] += x_need
    else:
        y_need = (x_length - y_length) / 2
        spatial_limit["y_min"] -= y_need
        spatial_limit["y_max"] += y_need
    img_inverted_matrix, normalized_depth_img = ray_tracing_depth_map(pc_mesh, side_range=(
        spatial_limit["x_min"], spatial_limit["x_max"]), fwd_range=(spatial_limit["y_min"], spatial_limit["y_max"]),
                                                                      res=res,
                                                                      z_max_camera=spatial_limit["z_max"] + 0.5,
                                                                      max_z_distance=spatial_limit["z_max"] -
                                                                                     spatial_limit["z_min"] + 1)
    return img_inverted_matrix, normalized_depth_img


if __name__ == '__main__':
    stl_file_path = "notebooks/character_3d_aligned/02801_mk30/stl/0.stl"
    _, img = convert_pc_to_depth_map(stl_file_path, res=(512, 512, 255))
    import matplotlib.pyplot as plt
    plt.imshow(img, cmap='gray')
    plt.show()
