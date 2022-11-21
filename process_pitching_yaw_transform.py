from pointcloud_to_depth import convert_pc_to_depth_map
from model_transform import raw_pitch_transform_3d_points
from parsing import get_point_from_via_file
import argparse
from pathlib import Path
from model_transform_utils import read_stl_file, get_border_points_from_triangle_mesh
import cv2
from tqdm.auto import tqdm
import gc
import open3d as o3d
import numpy as np


def parse_aug():
    parser = argparse.ArgumentParser(prog='Generate gt depth maps')
    parser.add_argument('-woodblock', '--woodblock_path', help='path to whole woodblock dir/file')
    parser.add_argument('-floor', '--floor_path', help='path to the floor of woodblock dir/file')
    parser.add_argument('-surface', '--surface_path', help='path to the surface of woodblock dir/file')
    parser.add_argument('-border', '--border_path',
                        help='via file that contains the border points of horizontal line of woodblock')
    parser.add_argument('-woodblock_dest', '--woodblock_dest',
                        help='path to the directory to save the woodblock result')
    parser.add_argument("-mirror", "--mirror", action="store_true", help='mirror Oz and Ox')
    parser.add_argument("-matrix_z", "--matrix_z_path", help="convert border depth point to 3D point")
    parser.add_argument('-depth_dest', '--depth_dest', help='path to the directory to save the depth map result')
    parser.add_argument('-inverted_matrix', '--inverted_matrix_path',
                        help='path to the directory to save the inverted matrix that helps map anno depth to 3D model')
    parser.add_argument('-surface_xyz', '--surface_xyz',
                        help='path to the directory to save the surface stl aligned')
    args = parser.parse_args()
    return args


def main():
    args = parse_aug()

    if Path(args.woodblock_path).is_file():
        woodblock_path_list = [Path(args.woodblock_path)]
        floor_path_list = [Path(args.floor_path)]
        surface_path_list = [Path(args.surface_path)]
        border_path_list = [Path(args.border_path)]
        matrix_z_list = [Path(args.matrix_z_path)]
    else:
        woodblock_path_list = list(Path(args.woodblock_path).glob("*.stl"))
        floor_path_list = list(Path(args.floor_path).glob("*.stl"))
        surface_path_list = list(Path(args.surface_path).glob("*.stl"))
        border_path_list = list(Path(args.border_path).glob("*.json"))
        matrix_z_list = list(Path(args.matrix_z_path).glob("*.npy"))

        woodblock_path_list.sort(key=lambda file_name: file_name.stem)
        floor_path_list.sort(key=lambda file_name: file_name.stem)
        surface_path_list.sort(key=lambda file_name: file_name.stem)
        border_path_list.sort(key=lambda border_name: border_name.stem)
        matrix_z_list.sort(key=lambda matrix_z: matrix_z.stem)

        assert len(woodblock_path_list) == len(floor_path_list), \
            f"Num files of woodblock: {woodblock_path_list} is not equal to num file of surface: {floor_path_list}"
    for index, woodblock_path in tqdm(enumerate(woodblock_path_list)):
        try:
            floor_path = floor_path_list[index]
            surface_path = surface_path_list[index]
            border_path = border_path_list[index]

            pc_floor_points = read_stl_file(floor_path)
            pc_surface_points = read_stl_file(surface_path)
            pc_woodblock_points = read_stl_file(woodblock_path)
            matrix_z = np.load(str(matrix_z_list[index]))
            [_, border_points] = get_point_from_via_file(border_path, keyword='whole')
            border_points = np.hstack((border_points, np.zeros((border_points.shape[0], 1))))
            pc_woodblock_points, pc_surface_points = raw_pitch_transform_3d_points(pc_woodblock_points, pc_surface_points, pc_floor_points, border_points,
                                                                matrix_z, args.mirror)
            normalized_inverted_matrix, normalized_depth_img = convert_pc_to_depth_map(pc_woodblock_points)

            cv2.imwrite(str(Path(args.depth_dest) / f'{woodblock_path.stem}_xyz.png'), normalized_depth_img)
            o3d.io.write_triangle_mesh(str(Path(args.woodblock_dest) / f'{woodblock_path.stem}_xyz.stl'), pc_woodblock_points)
            o3d.io.write_triangle_mesh(str(Path(args.surface_xyz) / f'{surface_path.stem}_xyz.stl'), pc_surface_points)

            with open(str(Path(args.inverted_matrix_path)/f'{woodblock_path.stem}_xyz.npy'), 'wb') as f:
                np.save(f, normalized_inverted_matrix)
            del pc_woodblock_points
            del normalized_depth_img
            del pc_surface_points
            gc.collect()
        except Exception as e:
            print(e)
            print(woodblock_path.name)


if __name__ == '__main__':
    main()



