from crop_utils import *
from model_transform_utils import read_stl_file
from parsing import get_point_from_via_file, parse_bboxes_list_from_sino_nom_anno_file
import argparse
import os
import cv2
from config import args
from model_transform_utils import get_border_points_from_triangle_mesh


def parse_aug():
    parser = argparse.ArgumentParser(prog='Generate gt depth maps')
    parser.add_argument('-woodblock', '--woodblock_path', help='path to whole woodblock dir/file')
    parser.add_argument('-surface', '--surface_path', help='path to the surface of woodblock dir/file')
    parser.add_argument('-floor', '--floor', help='path to the surface of woodblock dir/file')
    parser.add_argument('-scan_img', '--scan_img', help='path to scan 2d')

    parser.add_argument('-pitch_map', '--pitch_map',
                        help='via anno that contains correspond pairs of 2d points and 3d points')
    parser.add_argument('-border', '--border_path', help='path to border point of horizontal line of woodblock')

    parser.add_argument('-sina_nom_anno', '--sina_nom_anno', help='path to sina-nom annotation')
    parser.add_argument('-inverted_matrix', '--inverted_matrix', help='the matrix to convert back to 3D from depth')

    parser.add_argument('-character_dest', '--character_dest',
                        help='path to the directory to save the aligned character result')

    parser.add_argument("-mirror", "--mirror", action="store_true", help='mirror Oz and Ox')
    parser.add_argument("-order", "--order_border", action="store_true", help='swap border points')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    Path(os.path.join(args.character_dest, Path(args.sina_nom_anno).stem, 'stl')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.character_dest, Path(args.sina_nom_anno).stem, 'depth')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.character_dest, Path(args.sina_nom_anno).stem, 'print')).mkdir(parents=True, exist_ok=True)

    if Path(args.woodblock_path).is_file():
        woodblock_path_list = [Path(args.woodblock_path)]
        floor_path_list = [Path(args.floor)]
        surface_path_list = [Path(args.surface_path)]
        pitch_map_list = [Path(args.pitch_map)]
        border_path_list = [Path(args.border_path)]
        inverted_map_list = [Path(args.inverted_matrix)]
        sina_nom_bbox_list = [Path(args.sina_nom_anno)]
        scan_img_list = [Path(args.scan_img)]

    else:
        woodblock_path_list = list(Path(args.woodblock_path).glob("*.stl"))
        floor_path_list = list(Path(args.floor).glob("*.stl"))
        surface_path_list = list(Path(args.surface_path).glob("*.stl"))
        pitch_map_list = list(Path(args.pitch_map).glob("*.json"))
        border_path_list = list(Path(args.border_path).glob("*.stl"))
        inverted_map_list = list(Path(args.inverted_matrix).glob("*.npy"))
        sina_nom_bbox_list = list(Path(args.sina_nom_anno).glob("*.json"))
        scan_img_list = list(Path(args.scan_img).glob("*.jpg"))

        woodblock_path_list.sort(key=lambda file_name: file_name.stem)
        floor_path_list.sort(key=lambda file_name: file_name.stem)
        surface_path_list.sort(key=lambda file_name: file_name.stem)
        pitch_map_list.sort(key=lambda file_name: file_name.stem)
        border_path_list.sort(key=lambda file_name: file_name.stem)
        inverted_map_list.sort(key=lambda file_name: file_name.stem)
        sina_nom_bbox_list.sort(key=lambda file_name: file_name.stem)
        scan_img_list.sort(key=lambda file_name: file_name.stem)

    for index, woodblock_path in tqdm(enumerate(woodblock_path_list)):
        inverted_matrix = np.load(str(inverted_map_list[index]))
        woodblock_points = read_stl_file(woodblock_path)
        woodblock_floor_points = read_stl_file(floor_path_list[index])
        surface_points = read_stl_file(surface_path_list[index])
        border_path = border_path_list[index]
        [point_2d_list, point_depth_list] = get_point_from_via_file(pitch_map_list[index], keyword='whole')
        border_mesh_points = read_stl_file(border_path)
        border_points = get_border_points_from_triangle_mesh(border_mesh_points, args.order_border)
        bboxes_2d_list = parse_bboxes_list_from_sino_nom_anno_file(sina_nom_bbox_list[index])

        aligned_3d_character_list = crop_3d_characters(woodblock_points,
                                                        woodblock_floor_points,
                                                        surface_points,
                                                        bboxes_2d_list,
                                                        point_2d_list,
                                                        point_depth_list,
                                                        border_points,
                                                        inverted_matrix,
                                                        args.mirror)

        for w_index, aligned_3d_character in enumerate(aligned_3d_character_list):

            o3d.io.write_triangle_mesh(os.path.join(args.character_dest, Path(args.sina_nom_anno).stem, f'stl/{w_index}.stl'), aligned_3d_character)

        depth_img_list, _ = get_character_depth_imgs(aligned_3d_character_list)

        for c_index, character_depth in enumerate(depth_img_list):
            cv2.imwrite(os.path.join(args.character_dest, Path(args.sina_nom_anno).stem, f'depth/{c_index}.png'), cv2.flip(character_depth, 1))

        print_img = cv2.imread(str(scan_img_list[index]))
        for s_index, bbox in enumerate(bboxes_2d_list):
            character_img = crop_2d_img(print_img, bbox)
            cv2.imwrite(str(os.path.join(args.character_dest, Path(args.sina_nom_anno).stem, f'print/{s_index}.png')), character_img)


#