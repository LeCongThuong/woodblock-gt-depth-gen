import numpy as np
from skimage import transform


def mapping(bboxes_list, np_src_point_list, np_dest_point_list):
    tform = transform.estimate_transform('projective', np_src_point_list, np_dest_point_list)
    np_bboxes = np.asarray(bboxes_list)
    np_dest_bboxes = []
    for poly_points in np_bboxes:
        np_dest_bboxes.append(tform(poly_points))
    return np_dest_bboxes


def mapping_depth_point_to_3d_point(depth_poly_list, inverted_matrix):
    homo_depth_poly_list = []
    for depth_poly_points in depth_poly_list:
        one_z = np.ones((depth_poly_points.shape[0], 1))
        point_3d = np.c_[depth_poly_points, one_z, one_z]
        homo_depth_poly_list.append(point_3d)
    woodblock_3d_points_list = []
    for homo_depth_poly_point in homo_depth_poly_list:
        point_3d = inverted_matrix@homo_depth_poly_point.T
        point_3d = point_3d.T
        woodblock_3d_points_list.append(point_3d)
    return woodblock_3d_points_list


def get_3d_points_by_mapping_2d_3d_points(bboxes_2d_list, inverted_matrix, point_2d_list, point_depth_list):
    """
    Mapping bounding boxes of 2D scan to bounding boxes of 3D model. Firstly, mapping bboxes 2d list of 2D scan to
    depth map image by using registered points (point_2d_list, point_depth_list). Secondly, mapping the bboxes of depth
    map images to 3D by using inverted_matrix
    :param bboxes_2d_list: bboxes 2d of scan images
    :param inverted_matrix: matrix that can be used to get points of 3D from points of depth map
    :param point_2d_list: registered 2d points
    :param point_depth_list: registered depth points
    :return:
    """
    bboxes_depth_list = mapping(bboxes_2d_list, point_2d_list, point_depth_list)
    points_3d_list = mapping_depth_point_to_3d_point(bboxes_depth_list, inverted_matrix)
    return points_3d_list



