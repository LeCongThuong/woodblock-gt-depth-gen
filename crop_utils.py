import open3d as o3d
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from math import sqrt
from model_transform_utils import get_surface_equation_coeffs
from pointcloud_to_depth import convert_pc_to_depth_map
import cv2
from model_transform import raw_pitch_transform_3d_points
from mapping_utils import get_3d_points_by_mapping_2d_3d_points


def get_bboxes_bound(bboxes_points_list, z_min_bound=-20, z_max_bound=20):
    character_rect_bound_list = []
    for index, poly_points in enumerate(tqdm(bboxes_points_list)):
        min_x, max_x = np.min(poly_points[:, 0]), np.max(poly_points[:, 0])
        min_y, max_y = np.min(poly_points[:, 1]), np.max(poly_points[:, 1])
        min_bound = np.asarray([min_x, min_y, z_min_bound]).reshape(3, 1)
        max_bound = np.asarray([max_x, max_y, z_max_bound]).reshape(3, 1)
        character_rect_bound_list.append(np.c_[min_bound, max_bound])
    return character_rect_bound_list


def crop_3d_characters(woodblock_points, bboxes_points_list, z_min_bound=-20, z_max_bound=20):
    character_point_list = []
    character_rect_bound_list = get_bboxes_bound(bboxes_points_list, z_min_bound, z_max_bound)
    for index, poly_points in enumerate(tqdm(bboxes_points_list)):
        character_rect_bound = character_rect_bound_list[index]
        aligned_bb = o3d.geometry.AxisAlignedBoundingBox(character_rect_bound[0], character_rect_bound[1])
        character_pc = woodblock_points.crop(aligned_bb)
        character_pc = o3d.geometry.TriangleMesh.compute_triangle_normals(character_pc)
        character_pc = character_pc.remove_duplicated_vertices()
        # np_character_pc = np.asarray(character_pc.vertices)
        character_point_list.append(character_pc)
    return character_point_list


def crop_polygon_3d_characters(woodblock_points, bboxes_points_list, z_min_bound=-25, z_max_bound=25):
    character_point_list = []
    for index, poly_points in enumerate(tqdm(bboxes_points_list)):
        bounding_polygon = poly_points[:, :3].astype("float64")
        vol = o3d.visualization.SelectionPolygonVolume()
        vol.orthogonal_axis = "Z"
        vol.axis_max = z_max_bound
        vol.axis_min = z_min_bound
        vol.bounding_polygon = o3d.utility.Vector3dVector(bounding_polygon)
        character_pc = vol.crop_triangle_mesh(woodblock_points)
        character_pc = o3d.geometry.TriangleMesh.compute_triangle_normals(character_pc)
        character_pc = character_pc.remove_duplicated_vertices()
        character_point_list.append(character_pc)
    return character_point_list


def calculate_direction_normal_vector(C, points):
    gradient_of_x = C[1] + C[4] * points[:, 1] + 2*C[5]*points[:, 0]
    gradient_of_y = C[2] + C[4] * points[:, 0] + 2*C[6]*points[:, 1]
    minus_ones = -np.ones_like(gradient_of_x)
    gradient = np.c_[gradient_of_x, gradient_of_y, minus_ones]
    return np.mean(gradient, axis=0)


def get_position_grid(limit_bounding):
    limit_bounding = limit_bounding.astype(np.int16)
    x = np.arange(limit_bounding[0][0], limit_bounding[0][1])
    y = np.arange(limit_bounding[1][0], limit_bounding[1][1])
    xx, yy = np.meshgrid(x, y)
    position_grid = np.column_stack((xx.ravel(), yy.ravel()))
    return position_grid


def get_all_normal_vectors(character_rect_bound_list, plane_coff_2):
    normal_vector_list = []
    for character_rect_bound in character_rect_bound_list:
        position_grid = get_position_grid(character_rect_bound)
        normal_vector = calculate_direction_normal_vector(plane_coff_2, position_grid)
        normal_vector_list.append(normal_vector)
    return normal_vector_list


def do_character_transform(np_pc_points, transform_matrix):
    ones = np.ones((np_pc_points.shape[0], 1))
    homo_np_pc_points = np.c_[np_pc_points, ones]
    inverted_transform_matrix = np.linalg.inv(transform_matrix)
    homo_pc_3d = inverted_transform_matrix @ homo_np_pc_points.T
    # homo_pc_3d = transform_matrix@(homo_np_pc_points.T)
    homo_pc_3d = homo_pc_3d.T[:, :3]

    mean_vector = np.mean(homo_pc_3d, axis=0)
    # mean_vector[2] = 0
    homo_pc_3d = homo_pc_3d - mean_vector
    return homo_pc_3d


def calc_u1(a, b, c):
    return b / sqrt(a * a + b * b)


def calc_u2(a, b, c):
    return -a / sqrt(a * a + b * b)


def calc_sin_phi(a, b, c):
    return sqrt((a * a + b * b) / (a * a + b * b + c * c))


def calc_cos_phi(a, b, c):
    return c / sqrt(a * a + b * b + c * c)


def get_rotation_matrix_of_character(normalized_vector):
    a, b, c = normalized_vector
    if c < 0:
        a, b, c = -a, -b, -c
    cos_phi = calc_cos_phi(a, b, c)
    sin_phi = calc_sin_phi(a, b, c)
    u1 = calc_u1(a, b, c)
    u2 = calc_u2(a, b, c)
    rot_matrix = np.array([
        [cos_phi + u1 * u1 * (1 - cos_phi), u1 * u2 * (1 - cos_phi), u2 * sin_phi, 0],
        [u1 * u2 * (1 - cos_phi), cos_phi + u2 * u2 * (1 - cos_phi), -u1 * sin_phi, 0],
        [-u2 * sin_phi, u1 * sin_phi, cos_phi, 0],
        [0, 0, 0, 1]])
    return rot_matrix


def transform_characters(pc_points, normalized_vector):
    np_pc_points = np.asarray(pc_points.vertices)
    transform_matrix = get_rotation_matrix_of_character(normalized_vector)
    homo_pc_3d = do_character_transform(np_pc_points, transform_matrix)
    pc_points.vertices = o3d.utility.Vector3dVector(homo_pc_3d)
    pc_points = o3d.geometry.TriangleMesh.compute_triangle_normals(pc_points)
    pc_points.remove_duplicated_vertices()
    return pc_points


def get_aligned_3d_characters(character_point_list, normal_vector_list):
    aligned_pc_point_list = []
    for index, character_point in enumerate(tqdm(character_point_list)):
        pc_points = transform_characters(character_point, normal_vector_list[index])
        aligned_pc_point_list.append(pc_points)
    return aligned_pc_point_list


def crop_3d_characters(woodblock_points, woodblock_floor_points, character_surface_points, bboxes_2d_list, point_2d_list, point_depth_list, inverted_matrix, z_min_bound=-25, z_max_bound=25):
    """
    Crop 3D characters from a whole woodblock. To do that:
    Firstly, get 3D points bounding boxes by mapping from 2D bboxes
    Secondly, crop vanilla characters using 3D points bounding boxes
    Thirdly, get aligned characters:
        + Get 2-degree surface equation of woodblock
        + Get normal vectors of each character
        +  Pitch and raw rotation to get aligned character
    :param woodblock_points: whole woodblock
    :param woodblock_floor_points: points on floor of woodblock
    :param character_surface_points: points on surface of characters
    :param bboxes_2d_list: bounding boxes on 2D scan images
    :param point_2d_list:  registered points on 2D scan images
    :param point_depth_list: registered points on depth images
    :param inverted_matrix: matrix to get back points on 3D model from points on depth map
    :param z_min_bound: min bounding of z axis
    :param z_max_bound: max bounding of z axis
    :return: 3D aligned mesh of all 3D characters
    """
    bboxes_3d_list = get_3d_points_by_mapping_2d_3d_points(bboxes_2d_list, inverted_matrix, point_2d_list, point_depth_list)
    character_rect_bound_list = get_bboxes_bound(bboxes_3d_list, z_min_bound, z_max_bound)

    character_surface_points = raw_pitch_transform_3d_points(woodblock_floor_points, character_surface_points, point_depth_list)
    surface_2d_coeffs = get_surface_equation_coeffs(np.asarray(character_surface_points.vertices), order=2)
    character_point_list = crop_polygon_3d_characters(woodblock_points, bboxes_3d_list, z_min_bound, z_max_bound)

    normal_vector_list = get_all_normal_vectors(character_rect_bound_list, surface_2d_coeffs)

    aligned_3d_character_list = get_aligned_3d_characters(character_point_list, normal_vector_list)

    del character_point_list
    del normal_vector_list
    del character_rect_bound_list
    return aligned_3d_character_list


def get_character_depth_imgs(character_3d_list, res=(512, 512, 255)):
    depth_map_img_list = []
    img_inverted_matrix_list = []
    for character_3d in tqdm(character_3d_list):
        img_inverted_matrix, normalized_depth_img = convert_pc_to_depth_map(character_3d, res=res)
        depth_map_img_list.append(normalized_depth_img)
        img_inverted_matrix_list.append(img_inverted_matrix)
    return depth_map_img_list, img_inverted_matrix_list


def resize_image_to_square(image, dst=512, color=(255, 255, 255)):
    """
    Resize with padding, first resize image such that max edge get dst, then pad 2 sides of the other edge to dest
    :param image: cv2 image
    :param dst: target size
    :param color: constant pad value
    :return: square image
    """
    desired_size = dst

    im = image
    height, width = im.shape[:2]  # old_size is in (height, width) format

    delta = abs(height - width)
    if height > width:
        top, bottom, left, right = 0, 0, delta // 2, delta - (delta // 2)
    else:
        top, bottom, left, right =  delta // 2, delta - (delta // 2), 0, 0
    color = color
    new_img = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    new_img = cv2.resize(new_img, (dst, dst))
    return new_img


def crop_2d_img(img, pts):
    pts = np.array(pts, dtype=np.int32)
    ## (1) Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    croped = img[y:y+h, x:x+w].copy()

    ## (2) make mask
    pts = pts - pts.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)

    ## (4) add the white background
    bg = np.ones_like(croped, np.uint8)*255
    cv2.bitwise_not(bg, bg, mask=mask)
    dst2 = bg + dst
    dst2 = resize_image_to_square(dst2)
    return dst2

