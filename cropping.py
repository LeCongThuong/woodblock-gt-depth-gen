from crop_utils import *
from mapping_utils import get_3d_points_by_mapping_2d_3d_points
from model_transform import raw_pitch_transform_3d_points
from model_transform_utils import get_surface_equation_coeffs


def crop_characters(woodblock_points, woodblock_floor_points, character_surface_points, depth_border_points, bboxes_2d_list, point_2d_list, point_depth_list, inverted_matrix, z_min_bound=-25, z_max_bound=25):
    bboxes_3d_list = get_3d_points_by_mapping_2d_3d_points(bboxes_2d_list, inverted_matrix, point_2d_list, point_depth_list)
    character_rect_bound_list = get_bboxes_bound(bboxes_3d_list, z_min_bound, z_max_bound)

    character_surface_points = raw_pitch_transform_3d_points(woodblock_floor_points, character_surface_points, depth_border_points)
    character_point_list = crop_polygon_3d_characters(woodblock_points, bboxes_3d_list, z_min_bound, z_max_bound)

    surface_2d_coeffs = get_surface_equation_coeffs(character_surface_points, order=2)
    normal_vector_list = get_all_normal_vectors(character_rect_bound_list, surface_2d_coeffs)

    aligned_3d_character_list = get_aligned_3d_characters(character_point_list, normal_vector_list)

    del character_point_list
    del normal_vector_list
    del character_rect_bound_list
    return aligned_3d_character_list

