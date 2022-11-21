import argparse
from process_pitching_yaw_transform import process_pitching_yaw_transform
import os
from pathlib import Path
from tqdm import tqdm


def parse_aug():
    parser = argparse.ArgumentParser(prog='Generate gt depth maps')
    parser.add_argument('-id_path', '--id_path', help='path to txt file listing id of all the woodblocks')
    parser.add_argument('-raw', '--raw', help='path to the directory containing raw data needed to process')
    parser.add_argument("-mirror", "--mirror_path", help='file path containing wb id need to mirror Oz and Ox')
    parser.add_argument('-interim', '--interim', help='path to the directory of the interim output')
    parser.add_argument('-output', '--output', help='path to the directory of output')
    parser.add_argument('-error', "--error_path", help="path to error file contains all error woodblock ids")
    args = parser.parse_args()
    return args


def process_path(woodblock_id, raw_path, interim_dir, output_path, wb_key="whole", sf_key="surface", fl_key='floor',
                 br_key='border', mat_key='matrix_z', depth_key="depth_z"):
    woodblock_id_dir = os.path.join(raw_path, woodblock_id)
    file_path_list = list(Path(woodblock_id_dir).glob("*"))
    for file_path in file_path_list:
        file_name = file_path.stem
        if sf_key in file_name:
            surface_path = str(file_path)
        if wb_key in file_name:
            woodblock_path = str(file_path)
        if fl_key in file_name:
            floor_path = str(file_path)

    interim_wb_id_dir = os.path.join(interim_dir, woodblock_id)
    file_path_list = list(Path(interim_wb_id_dir).glob("*"))
    for file_path in file_path_list:
        file_name = file_path.stem
        if br_key in file_name:
            depth_border_path = str(file_path)
        if mat_key in file_name:
            matrix_z_path = str(file_path)
        if depth_key in file_name:
            depth_z_path = str(file_path)

    surface_xyz_path = os.path.join(interim_wb_id_dir, f'surface_xyz_{woodblock_id}.stl')

    woodblock_id_dest = os.path.join(output_path, woodblock_id)
    Path(woodblock_id_dest).mkdir(exist_ok=True, parents=True)
    matrix_xyz_path = os.path.join(woodblock_id_dest, f'matrix_xyz_{woodblock_id}.npy')
    depth_xyz_path = os.path.join(woodblock_id_dest, f'depth_xyz_{woodblock_id}.png')
    dest_woodblock_xyz_path = os.path.join(woodblock_id_dest, f"whole_xyz_{woodblock_id}.stl")

    return woodblock_path, surface_path, floor_path, depth_border_path, matrix_z_path, depth_z_path, surface_xyz_path, \
           matrix_xyz_path, depth_xyz_path, dest_woodblock_xyz_path


def run():
    args = parse_aug()
    with open(args.id_path, 'r') as f:
        content = f.read()
    wb_id_list = content.split("\n")
    with open(args.mirror_path, 'r') as f:
        content = f.read()
    mirror_id_wb_list = content.split("\n")
    for index, wb_id in tqdm(enumerate(wb_id_list)):
        try:
            woodblock_path, surface_path, floor_path, depth_border_path, matrix_z_path, depth_z_path, surface_xyz_path, \
            matrix_xyz_path, depth_xyz_path, dest_woodblock_xyz_path = process_path(wb_id, args.raw, args.interim,
                                                                                    args.output)
            do_mirror = True if wb_id in mirror_id_wb_list else False
            process_pitching_yaw_transform(woodblock_path, floor_path, surface_path, depth_border_path, matrix_z_path, do_mirror,
                                   surface_xyz_path, matrix_xyz_path, depth_xyz_path, dest_woodblock_xyz_path)
        except Exception as e:
            print(e)
            with open(args.error_path, "a") as f:
                f.write(f"{wb_id}\n")


if __name__ == '__main__':
    run()
