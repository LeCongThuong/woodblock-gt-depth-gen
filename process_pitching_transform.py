from pointcloud_to_depth import convert_pc_to_depth_map
from model_transform import pitch_transform_3d_points
import argparse
from pathlib import Path
from model_transform_utils import read_stl_file
import cv2
from tqdm.auto import tqdm
import gc
import open3d as o3d
from model_transform_utils import preprocess_pc


def parse_aug():
    parser = argparse.ArgumentParser(prog='Generate gt depth maps')
    parser.add_argument('-woodblock', '--woodblock_path', help='path to whole woodblock dir/file')
    parser.add_argument('-surface', '--surface_path', help='path to the surface of woodblock dir/file')
    parser.add_argument('-floor', '--floor_path', help='path to the floor of woodblock dir/file')
    # parser.add_argument('-woodblock_dest', '--woodblock_dest', help='path to the directory to save the woodblock result')
    parser.add_argument("-upside", "--upside", action="store_true", help='rotate woodblock around Oy, for beneath surface')
    parser.add_argument('-depth_dest', '--depth_dest', help='path to the directory to save the depth map result')
    args = parser.parse_args()
    return args


def take_name(file_path):
    return file_path.stem


def main():
    args = parse_aug()

    if Path(args.woodblock_path).is_file():
        woodblock_path_list = [Path(args.woodblock_path)]
        surface_path_list = [Path(args.surface_path)]
        floor_path_list = [Path(args.floor_path)]
    else:
        woodblock_path_list = list(Path(args.woodblock_path).glob("*.stl"))
        surface_path_list = list(Path(args.surface_path).glob("*.stl"))
        floor_path_list = list(Path(args.floor_path).glob("*.stl"))
        woodblock_path_list.sort(key=lambda file_name: file_name.stem)
        surface_path_list.sort(key=lambda file_name: file_name.stem)
        floor_path_list.sort(key=lambda file_name: file_name.stem)
    assert len(woodblock_path_list) == len(surface_path_list), f"Num files of woodblock: {woodblock_path_list} is not equal to num file of surface: {surface_path_list}"
    for index, woodblock_path in tqdm(enumerate(woodblock_path_list)):
        # try:
        surface_path = surface_path_list[index]
        floor_path = floor_path_list[index]
        pc_surface_points = read_stl_file(surface_path)
        pc_floor_points = read_stl_file(floor_path)
        pc_woodblock_points = read_stl_file(woodblock_path)
        # if args.upside:
        #     pc_woodblock_points, pc_surface_points, pc_floor_points = preprocess_pc(pc_woodblock_points, pc_surface_points, pc_floor_points)
        #     # print(f'./notebooks/woodblock_z/{Path(woodblock_path).stem}_z.stl')
        # o3d.io.write_triangle_mesh(str('/mnt/hdd/thuonglc/mocban/woodblock-gt-depth-gen/notebooks/woodblock_z/02801_mk29_z.stl'), pc_woodblock_points)
        # o3d.io.write_triangle_mesh('/mnt/hdd/thuonglc/mocban/woodblock-gt-depth-gen/notebooks/floor_z/02801_mk29_z.stl', pc_floor_points)
        # o3d.io.write_triangle_mesh('/mnt/hdd/thuonglc/mocban/woodblock-gt-depth-gen/notebooks/surface_z/02801_mk29_z.stl', pc_surface_points)

        pc_woodblock_points = pitch_transform_3d_points(pc_floor_points, pc_woodblock_points)

        _, normalized_depth_img = convert_pc_to_depth_map(pc_woodblock_points)
        cv2.imwrite(str(Path(args.depth_dest) / f'{woodblock_path.stem}_z.png'), normalized_depth_img)
        del pc_woodblock_points
        del normalized_depth_img
        gc.collect()
        # except Exception as e:
        #     print(e)
        #     print(woodblock_path.name)


if __name__ == '__main__':
    main()



