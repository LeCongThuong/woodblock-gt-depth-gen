from crop_utils import *
from mapping_utils import get_3d_points_by_mapping_2d_3d_points
from model_transform import raw_pitch_transform_3d_points
from model_transform_utils import get_surface_equation_coeffs, read_stl_file
from parsing import get_point_from_via_file, parse_bboxes_list_from_sino_nom_anno_file
import argparse
import os
import cv2


def parse_aug():
    parser = argparse.ArgumentParser(prog='Generate gt depth maps')
    parser.add_argument('-woodblock', '--woodblock_path', help='path to whole woodblock dir/file')
    parser.add_argument('-character_surface', '--character_surface', help='path to the surface of woodblock dir/file')
    parser.add_argument('-floor', '--floor', help='path to the surface of woodblock dir/file')

    parser.add_argument('-pitch_map', '--pitch_map', help='via anno that contains correspond pairs of 2d points and 3d points')
    parser.add_argument('-sina_nom_anno', '--sina_nom_anno', help='path to sina-nom annotation')
    parser.add_argument('-inverted_matrix', '--inverted_matrix', help='the matrix to convert back to 3D from depth')

    parser.add_argument('-character_dest', '--character_dest', help='path to the directory to save the aligned character result')


    args = parser.parse_args()
    return args


def crop_characters(woodblock_points, woodblock_floor_points, character_surface_points, bboxes_2d_list, point_2d_list, point_depth_list, inverted_matrix, z_min_bound=-25, z_max_bound=25):
    bboxes_3d_list = get_3d_points_by_mapping_2d_3d_points(bboxes_2d_list, inverted_matrix, point_2d_list, point_depth_list)
    character_rect_bound_list = get_bboxes_bound(bboxes_3d_list, z_min_bound, z_max_bound)

    character_surface_points = raw_pitch_transform_3d_points(woodblock_floor_points, character_surface_points, point_depth_list)
    character_point_list = crop_polygon_3d_characters(woodblock_points, bboxes_3d_list, z_min_bound, z_max_bound)

    surface_2d_coeffs = get_surface_equation_coeffs(np.asarray(character_surface_points.vertices), order=2)
    normal_vector_list = get_all_normal_vectors(character_rect_bound_list, surface_2d_coeffs)

    aligned_3d_character_list = get_aligned_3d_characters(character_point_list, normal_vector_list)

    del character_point_list
    del normal_vector_list
    del character_rect_bound_list
    return aligned_3d_character_list


if __name__ == '__main__':
    # woodblock_path = 'notebooks/woodblock_xyz/whole_02801_mk29_xyz.stl'
    # woodblock_floor_path = "/media/hmi/Expansion/MOCBAN_TEST_OK/floor_woodblock_points/floor_02801_mk29.stl"
    # character_surface_path = "/media/hmi/Expansion/MOCBAN_TEST_OK/surface_woodblock_points/02801_woodblock_surface_1.stl"
    # pitch_mapping_path = "notebooks/pitch_map/pitch_map_02801_mk29.json"
    # sina_nom_bbox_path = "/media/hmi/Expansion/MOCBAN_TEST_OK/tt4_samples/tt4/02801_mk29/02801_mk29.json"
    # inverted_matrix_path = "notebooks/woodblock_inverted_matrix/whole_02801_mk29_xyz.npy"
    #
    # aligned_character_path = Path("notebooks/character_3d_aligned")/Path(sina_nom_bbox_path).stem
    args = parse_aug()
    Path(os.path.join(args.character_dest, Path(args.sina_nom_anno).stem, 'stl')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.character_dest, Path(args.sina_nom_anno).stem, 'depth')).mkdir(parents=True, exist_ok=True)

    if Path(args.woodblock_path).is_file():
        woodblock_path_list = [Path(args.woodblock_path)]
        floor_path_list = [Path(args.floor)]
        character_surface_list = [Path(args.character_surface)]
        pitch_map_list = [Path(args.pitch_map)]
        inverted_map_list = [Path(args.inverted_matrix)]
        sina_nom_bbox_list = [Path(args.sina_nom_anno)]

    else:
        woodblock_path_list = list(Path(args.woodblock_path).glob("*.stl"))
        floor_path_list = list(Path(args.floor).glob("*.stl"))
        character_surface_list = list(Path(args.character_surface).glob("*.stl"))
        pitch_map_list = list(Path(args.pitch_map).glob("*.json"))
        inverted_map_list = list(Path(args.inverted_matrix).glob("*.npy"))
        sina_nom_bbox_list = list(Path(args.sina_nom_anno).glob("*.json"))

        woodblock_path_list.sort(key=lambda file_name: file_name.stem)
        floor_path_list.sort(key=lambda file_name: file_name.stem)
        character_surface_list.sort(key=lambda file_name: file_name.stem)
        pitch_map_list.sort(key=lambda file_name: file_name.stem)
        inverted_map_list.sort(key=lambda file_name: file_name.stem)
        sina_nom_bbox_list.sort(key=lambda file_name: file_name.stem)

    for index, woodblock_path in tqdm(enumerate(woodblock_path_list)):
        inverted_matrix = np.load(str(inverted_map_list[index]))
        woodblock_points = read_stl_file(woodblock_path)
        woodblock_floor_points = read_stl_file(floor_path_list[index])
        character_surface_points = read_stl_file(character_surface_list[index])
        [point_2d_list, point_depth_list] = get_point_from_via_file(pitch_map_list[index], keyword='whole')
        bboxes_2d_list = parse_bboxes_list_from_sino_nom_anno_file(sina_nom_bbox_list[index])

        aligned_3d_character_list = crop_characters(woodblock_points,
                                                    woodblock_floor_points,
                                                    character_surface_points,
                                                    bboxes_2d_list,
                                                    point_2d_list,
                                                    point_depth_list,
                                                    inverted_matrix)

        for index, aligned_3d_character in enumerate(aligned_3d_character_list):
            o3d.io.write_triangle_mesh(f'{str(args.character_dest)}/stl/{index}.stl', aligned_3d_character)

        depth_img_list, _ = get_character_depth_imgs(aligned_3d_character_list)

        for index, character_depth in enumerate(depth_img_list):
            cv2.imwrite(f"{str(args.character_dest)}/depth/{index}.png", character_depth)

    # sample command
    # python3 cropping.py - woodblock "notebooks/woodblock_xyz/whole_02801_mk29_xyz.stl" -character_surface "/media/hmi/Expansion/MOCBAN_TEST_OK/surface_woodblock_points/surface_02801_mk29.stl" -floor "/media/hmi/Expansion/MOCBAN_TEST_OK/floor_woodblock_points/floor_02801_mk29.stl" -pitch_map
    # "notebooks/pitch_map/pitch_map_02801_mk29.json" -sina_nom_anno "/media/hmi/Expansion/MOCBAN_TEST_OK/tt4_samples/tt4/02801_mk29/02801_mk29.json" -inverted_matrix "notebooks/woodblock_inverted_matrix/whole_02801_mk29_xyz.npy" -character_dest "notebooks/character_3d_aligned"