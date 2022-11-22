from crop_utils import *
from model_transform_utils import read_stl_file
from parsing import get_point_from_via_file, parse_bboxes_list_from_sino_nom_anno_file
import argparse
import os
import cv2


def parse_aug():
    parser = argparse.ArgumentParser(prog='Generate gt depth maps')
    parser.add_argument('-id_path', '--id_path', help='path to txt file listing id of all the woodblocks')
    parser.add_argument('-raw', '--raw', help='path to the directory containing raw data needed to process')
    parser.add_argument('-interim', '--interim', help='path to the directory of the interim output')
    parser.add_argument('-output', '--output', help='path to the directory of output')
    parser.add_argument('-error', "--error_path", help="path to error file contains all error woodblock ids")
    args = parser.parse_args()
    return args


def process_path(woodblock_id, raw_dir, interim_dir, output_path, wb_key="whole_xyz", sf_key="surface_xyz",
                 mat_key='matrix_xyz', depth_key="depth_xyz", print_key="jpg", sino_key="json", register_key="mapping"):
    woodblock_id_dir = os.path.join(raw_dir, woodblock_id)
    file_path_list = list(Path(woodblock_id_dir).glob("*"))
    # print(file_path_list)
    for file_path in file_path_list:
        file_name = file_path.name
        if print_key in file_name:
            print_img_path = str(file_path)
        if sino_key in file_name:
            sino_nom_path = str(file_path)

    interim_wb_id_dir = os.path.join(interim_dir, woodblock_id)
    file_path_list = list(Path(interim_wb_id_dir).glob("*"))
    for file_path in file_path_list:
        file_name = file_path.name
        if sf_key in file_name:
            surface_xyz_path = str(file_path)
        if register_key in file_name:
            mapping_path = str(file_path)

    woodblock_id_dest = os.path.join(output_path, woodblock_id)
    file_path_list = list(Path(woodblock_id_dest).glob("*"))
    for file_path in file_path_list:
        file_name = file_path.name
        if mat_key in file_name:
            matrix_xyz_path = str(file_path)
        if depth_key in file_name:
            depth_xyz_path = str(file_path)
        if wb_key in file_name:
            woodblock_xyz_path = str(file_path)

    character_xyz_dir = os.path.join(woodblock_id_dest, "character_xyz")
    Path(character_xyz_dir).mkdir(exist_ok=True, parents=True)
    Path(os.path.join(character_xyz_dir, 'stl')).mkdir(exist_ok=True, parents=True)
    Path(os.path.join(character_xyz_dir, 'depth')).mkdir(exist_ok=True, parents=True)
    Path(os.path.join(character_xyz_dir, 'print')).mkdir(exist_ok=True, parents=True)
    return sino_nom_path, print_img_path, mapping_path, woodblock_xyz_path, surface_xyz_path, matrix_xyz_path, depth_xyz_path, character_xyz_dir


def run():
    args = parse_aug()
    with open(args.id_path, 'r') as f:
        content = f.read()
    wb_id_list = content.split("\n")

    for index, wb_id in tqdm(enumerate(wb_id_list)):
        sino_nom_path, print_img_path, mapping_path, woodblock_xyz_path, surface_xyz_path, matrix_xyz_path, depth_xyz_path, \
        character_xyz_dir = process_path(wb_id, args.raw, args.interim, args.output)
        aligned_matrix = np.load(str(matrix_xyz_path))
        [point_2d_list, point_depth_list] = get_point_from_via_file(mapping_path, keyword='depth_xyz')
        # print(point_2d_list, point_depth_list)
        woodblock_points = read_stl_file(woodblock_xyz_path)
        surface_points = read_stl_file(surface_xyz_path)
        bboxes_2d_list = parse_bboxes_list_from_sino_nom_anno_file(sino_nom_path)

        aligned_3d_character_list = crop_3d_characters(woodblock_points,
                                                        surface_points,
                                                        bboxes_2d_list,
                                                        point_2d_list,
                                                        point_depth_list,
                                                        aligned_matrix
                                                       )

        for w_index, aligned_3d_character in enumerate(aligned_3d_character_list):
            o3d.io.write_triangle_mesh(os.path.join(character_xyz_dir, f'stl/{w_index}.stl'), aligned_3d_character)

        depth_img_list, _ = get_character_depth_imgs(aligned_3d_character_list)

        for c_index, character_depth in enumerate(depth_img_list):
            cv2.imwrite(os.path.join(character_xyz_dir, f'depth/{c_index}.png'), character_depth)

        print_img = cv2.imread(print_img_path)
        for s_index, bbox in enumerate(bboxes_2d_list):
            character_img = crop_2d_img(print_img, bbox)
            cv2.imwrite(str(os.path.join(character_xyz_dir, f'print/{s_index}.png')), character_img)


if __name__ == '__main__':
    run()
