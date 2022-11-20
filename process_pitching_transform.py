from pointcloud_to_depth import convert_pc_to_depth_map
from model_transform import pitch_transform_3d_points
import argparse
from pathlib import Path
from model_transform_utils import read_stl_file
import cv2
from tqdm.auto import tqdm
import gc
import numpy as np


def parse_aug():
    parser = argparse.ArgumentParser(prog='Generate gt depth maps')
    parser.add_argument('-woodblock', '--woodblock_path', help='path to whole woodblock dir/file')
    parser.add_argument('-floor', '--floor_path', help='path to the floor of woodblock dir/file')
    parser.add_argument("-mirror", "--mirror", action="store_true", help='mirror Oz and Ox')
    parser.add_argument('-depth_dest', '--depth_dest', help='path to the directory to save the depth map result')
    parser.add_argument('-inverted_matrix', '--inverted_matrix_path', help='path to inverted matrix dir to save')
    args = parser.parse_args()
    return args


def take_name(file_path):
    return file_path.stem


def main():
    args = parse_aug()

    if Path(args.woodblock_path).is_file():
        woodblock_path_list = [Path(args.woodblock_path)]
        floor_path_list = [Path(args.floor_path)]
    else:
        woodblock_path_list = list(Path(args.woodblock_path).glob("*.stl"))
        floor_path_list = list(Path(args.floor_path).glob("*.stl"))
        woodblock_path_list.sort(key=lambda file_name: file_name.stem)
        floor_path_list.sort(key=lambda file_name: file_name.stem)
    assert len(woodblock_path_list) == len(floor_path_list), \
        f"Num files of woodblock: {woodblock_path_list} is not equal to num file of surface: {floor_path_list}"
    for index, woodblock_path in tqdm(enumerate(woodblock_path_list)):
        try:
            floor_path = floor_path_list[index]
            pc_floor_points = read_stl_file(floor_path)
            pc_woodblock_points = read_stl_file(woodblock_path)
            pc_woodblock_points = pitch_transform_3d_points(pc_floor_points, pc_woodblock_points, args.mirror)

            normalized_inverted_matrix, normalized_depth_img = convert_pc_to_depth_map(pc_woodblock_points)
            with open(str(Path(args.inverted_matrix_path) / f'{woodblock_path.stem}_z.npy'), 'wb') as f:
                np.save(f, normalized_inverted_matrix)
            cv2.imwrite(str(Path(args.depth_dest) / f'{woodblock_path.stem}_z.png'), normalized_depth_img)
            del pc_woodblock_points
            del normalized_depth_img
            gc.collect()
        except Exception as e:
            print("Error: ", e)
            print(woodblock_path.name)


if __name__ == '__main__':
    main()



