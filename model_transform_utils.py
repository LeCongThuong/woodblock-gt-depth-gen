import numpy as np
from math import sqrt
import open3d as o3d
import scipy
import math
import scipy.linalg


def read_stl_file(pc_path):
    pc_points = o3d.io.read_triangle_mesh(str(pc_path), print_progress=True)
    # np_pc_points = np.asarray(pc_points.vertices)
    return pc_points  # , np_pc_points


# def calc_sin_phi(a, b, c):
#     return sqrt((a*a + b*b) / (a*a + b*b + c*c))
#
#
# def calc_cos_phi(a, b, c):
#     return c / sqrt(a*a + b*b + c*c)
#
#
# def calc_u1(a, b, c):
#     return b / sqrt(a*a + b*b)
#
#
# def calc_u2(a, b, c):
#     return -a / sqrt(a*a + b*b)


def get_rot_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    print(a, b, v)
    if any(v):  # if not all zeros then
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        matrix_3x3 = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        matrix_zeros_3x1 = np.zeros((3, 1))
        matrix_zeros_ones_1x3 = np.array([0, 0, 0, 1]).reshape(1, 4)
        return np.vstack((np.hstack((matrix_3x3, matrix_zeros_3x1)), matrix_zeros_ones_1x3))
    else:
        return np.eye(4)  # cross of all zeros only occurs on identical directions


def get_pitch_rot_matrix(v1, v2=(0, 0, 1)):
    """
    Get rotation matrix to rotate 3D model around the cross product of 2 vectors (perhaps normal vector of 3D surface and unit vector (0, 0, 1))
    In other way, rotate the 3D model such that the surface of 3D model is parallel with the Oxy plane
    :param plane_coeff: coefficients of the 3D model surface equation
    :return: pitch matrix
    """
    rot_matrix = get_rot_matrix_from_vectors(v1, v2)
    return rot_matrix


def get_oz_trans_matrix(surface_coeffs):
    """
    Get Oz translation matrix: the matrix will lift up/down a 3D object such that
    after translation, the intersection point between the 3D (expansion) surface and Oz is (0, 0, 0)
    :param surface_coeffs:
    :return:
    """
    t_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -surface_coeffs[3]], [0, 0, 0, 1]])
    return t_matrix


# def get_rad(pivot_points, start_index_point=0, end_index_point=1, unit_vec=(1, 0)):
#     start_points, end_points = pivot_points[start_index_point], pivot_points[end_index_point]
#     woodblock_vec = end_points - start_points
#     counterwise_lock_angle = get_angle_between_two_vectors(unit_vec, woodblock_vec)
#     return counterwise_lock_angle

#
# def get_angle_between_two_vectors(vec_1, vec_2):
#     counterwise_lock_angle = np.arctan2(vec_1[0] * vec_2[1] - vec_1[1] * vec_2[0],
#                                          vec_1[0] * vec_2[0] + vec_1[1] * vec_2[1])
#     return counterwise_lock_angle


def get_yaw_rot_matrix(border_points, unit_vec=(1, 0, 0)):
    # rot_rad = get_rad(border_points, start_index_point, end_index_point, unit_vec=unit_vec)

    # yaw_matrix = np.asarray([[math.cos(rot_rad), -math.sin(rot_rad), 0, 0],
    #                          [math.sin(rot_rad), math.cos(rot_rad), 0, 0],
    #                          [0, 0, 1, 0],
    #                          [0, 0, 0, 1]])
    horizontal_vec = border_points[1] - border_points[0]  # if flip_vec else border_points[0] - border_points[1]
    horizontal_vec[2] = 0
    unit_vec = np.asarray(unit_vec)
    yaw_matrix = get_pitch_rot_matrix(horizontal_vec, unit_vec)
    return yaw_matrix


def get_3d_transform_matrix(surface_coeffs, border_point=None):
    """
    Get the translation matrix such that the 3D object can "lie" on Oxy (the surface of 3D object is Oxy).
    If border_point not None, rotate 3D object such that upper width vector is same direction as vector (1, 0) in the Oxy plane
    To do that:
        Firtly, lift 3D object to 3D surface intersect with Oz at the (0, 0, 0) point
        Secondly, rotate 3D object around a vector that is the cross product of normal vector of 3D surface and vector (0, 0, 1)
        Thirdly, rotate 3D object such that upper width vector is same direction as vector (1, 0) in the Oxy plane
    :param surface_coeffs: coefficient of 3D object surface
    :param border_point: 4 corner points (2D) of 3D object, 4 points (2D) around fish-tail
    :return: homo transform matrix (4x4)
    """
    normal_vector = surface_coeffs[:3]
    pitch_matrix = get_pitch_rot_matrix(normal_vector)
    # print(rot_matrix)
    oz_trans_matrix = get_oz_trans_matrix(surface_coeffs)
    # print(trans_matrix)
    transform_matrix = np.matmul(pitch_matrix, oz_trans_matrix)
    if border_point is not None:
        yaw_matrix = get_yaw_rot_matrix(border_point)
        print(yaw_matrix)
        transform_matrix = np.matmul(yaw_matrix, transform_matrix)
    return transform_matrix


def _transform_homo_points(np_points, transform_matrix):
    ones = np.ones((np_points.shape[0], 1))
    homo_np_points = np.c_[np_points, ones]
    homo_np_3d = transform_matrix @ (homo_np_points.T)
    homo_np_3d = homo_np_3d.T[:, :3]
    return homo_np_3d


def transform_points(np_pc_points, transform_matrix):
    """
    Transform 3D points with transform matrix.
    Converting pc_points into homo points and then transform.
    Turn upside if woodblock point surface is downside
    :param np_pc_points: (x, y, z)
    :param transform_matrix: 4x4 matrix
    :return:homo_pc_3d: (x, y, z) not (x, y, z, 1)
    """
    homo_pc_3d = _transform_homo_points(np_pc_points, transform_matrix)
    # if upside:
    #     # upsiding if front face is downside
    #     homo_pc_3d[:, 2] = - homo_pc_3d[:, 2]
    #     homo_pc_3d[:, 0] = - homo_pc_3d[:, 0]

    # mean_vector = np.mean(homo_pc_3d, axis=0)
    # mean_vector[2] = 0
    # homo_pc_3d = homo_pc_3d - mean_vector
    return homo_pc_3d


def preprocess_pc(woodblock_points, surface_points, floor_points, upside=True):
    def _recalculate_norm(pc_points):
        np_points = np.asarray(pc_points.vertices)
        pc_points.vertices = o3d.utility.Vector3dVector(np_points)
        pc_points = o3d.geometry.TriangleMesh.compute_triangle_normals(pc_points)
        pc_points.remove_duplicated_vertices()
        return pc_points
    if upside:
        woodblock_points = upflip_3d_points(woodblock_points)
        surface_points = upflip_3d_points(surface_points)
        floor_points = upflip_3d_points(floor_points)
    mean_vector = np.mean(np.asarray(woodblock_points.vertices), axis=0)
    woodblock_points.vertices = o3d.utility.Vector3dVector(np.asarray(woodblock_points.vertices) - mean_vector)
    surface_points.vertices = o3d.utility.Vector3dVector(np.asarray(surface_points.vertices) - mean_vector)
    floor_points.vertices = o3d.utility.Vector3dVector(np.asarray(floor_points.vertices) - mean_vector)
    return _recalculate_norm(woodblock_points), _recalculate_norm(surface_points), _recalculate_norm(floor_points)


# def transform_upside_points(np_pc_points, np_surface_points, np_floor_points, transform_matrix, upside=False):
#     """
#     Transform 3D points with transform matrix.
#     Converting pc_points into homo points and then transform.
#     Turn upside if woodblock point surface is downside
#     :param np_pc_points: (x, y, z)
#     :param transform_matrix: 4x4 matrix
#     :return:homo_pc_3d: (x, y, z) not (x, y, z, 1)
#     """
#
#     homo_pc_3d = _transform_homo_points(np_pc_points, transform_matrix)
#     np_surface_points = _transform_homo_points(np_surface_points)
#     np_floor_points = _transform_homo_points(np_floor_points)
#     mean_vector = np.mean(homo_pc_3d, axis=0)
#     if upside:
#         # upsiding if front face is downside
#         homo_pc_3d[:, 2] = - homo_pc_3d[:, 2]
#         homo_pc_3d[:, 0] = - homo_pc_3d[:, 0]
#         np_surface_points[:, 2] = - np_surface_points[:, 2]
#         np_surface_points[:, 0] = - np_surface_points[:, 0]
#         np_floor_points[:, 2] = - np_floor_points[:, 2]
#         np_floor_points[:, 0] = - np_floor_points[:, 0]
#         mean_vector = np.mean(homo_pc_3d, axis=0)
#
#     mean_vector[2] = 0
#     homo_pc_3d = homo_pc_3d - mean_vector
#     np_floor_points = np_floor_points - mean_vector
#     np_surface_points = np_surface_points - mean_vector
#     return homo_pc_3d, np_floor_points, np_surface_points


def upflip_3d_points(pc_points):
    np_points = np.asarray(pc_points.vertices)
    np_points[:, 2] = - np_points[:, 2]
    np_points[:, 0] = - np_points[:, 0]
    pc_points.vertices = o3d.utility.Vector3dVector(np_points)
    pc_points = o3d.geometry.TriangleMesh.compute_triangle_normals(pc_points)
    pc_points.remove_duplicated_vertices()
    return pc_points


def get_rotation_points(poly_points, M):
    homo_poly_points = np.column_stack((poly_points, np.ones((len(poly_points), 1))))
    transform_poly_points = np.dot(M, homo_poly_points.T)
    transform_poly_points = transform_poly_points.T
    return transform_poly_points


def get_surface_equation_coeffs(data, order=2):
    if order == 1:
        # best-fit linear plane z = a*x + b*y + c, where a, b, c are the coefficients that need to find
        # a = C[0], b = C[1], c = C[2]
        A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
        C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])   # coefficients
        return [C[0], C[1], -1, C[2]]
    elif order == 2:
        # best-fit quadratic curve z = a + b*x + c*y + d*x*y + e*x*x + f*y*y
        # a = C[0], b = C[1], c= C[2], d = C[3], e = C[4], f = C[5]
        A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2]**2]
        C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])
        return [C[0], C[1], C[2], -1, C[3], C[4], C[5]]


def get_border_points_from_triangle_mesh(border_tri_mesh):
    np_tri_mesh = np.asarray(border_tri_mesh.triangles)
    np_3d_points = np.asarray(border_tri_mesh.vertices)
    border_points = []
    for i in range(np_tri_mesh.shape[0]):
        mean_point = np.mean(np_3d_points[np_tri_mesh[i], :], axis=0).tolist()
        border_points.append(mean_point)
    border_points = np.asarray(border_points)
    return border_points


if __name__ == '__main__':
    v_2 = [1, 0, 0]
    v_1 = [0, 1, 0]
    print(get_rot_matrix_from_vectors(v_1, v_2))
