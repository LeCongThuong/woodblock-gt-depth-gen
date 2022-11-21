import argparse
from pathlib import Path
import os
from process_pitching_transform import process_pitching_transform
from tqdm import tqdm


def parse_aug():
    parser = argparse.ArgumentParser(prog='Generate gt depth maps')
    parser.add_argument('-id_path', '--id_path', help='path to txt file listing id of all the woodblocks')
    parser.add_argument('-raw', '--raw_data', help='path to the directory containing raw data needed to process')
    parser.add_argument("-mirror", "--mirror", action="store_true", help='mirror Oz and Ox')
    parser.add_argument('-dest', '--dest_dir', help='path to the directory to save the output')
    parser.add_argument('-error', "--error_path", help="path to error file contains all error woodblock ids")
    args = parser.parse_args()
    return args


def process_path(woodblock_id, data_path, dest_dir, wb_key="whole", fl_key='floor'):
    woodblock_id_dir = os.path.join(data_path, woodblock_id)
    file_path_list = list(Path(woodblock_id_dir).glob("*"))
    for file_path in file_path_list:
        file_name = file_path.stem
        if wb_key in file_name:
            woodblock_path = str(file_path)
        if fl_key in file_name:
            floor_path = str(file_path)
    woodblock_id_dest = os.path.join(dest_dir, woodblock_id)
    Path(woodblock_id_dest).mkdir(exist_ok=True, parents=True)
    matrix_z_path = os.path.join(woodblock_id_dest, f'matrix_z_{woodblock_id}.npy')
    depth_z_path = os.path.join(woodblock_id_dest, f'depth_z_{woodblock_id}.png')
    return woodblock_path, floor_path, depth_z_path, matrix_z_path


def run():
    args = parse_aug()
    with open(args.id_path, 'r') as f:
        content = f.read()
    wb_id_list = content.split("\n")

    for index, wb_id in tqdm(enumerate(wb_id_list)):
        try:
            woodblock_path, floor_path, depth_z_path, matrix_z_path = process_path(wb_id, args.raw_data, args.dest_dir)
            process_pitching_transform(woodblock_path, floor_path, args.mirror, depth_z_path, matrix_z_path)
        except Exception as e:
            print(e)
            with open(args.error_path, "a+") as f:
                f.write(f"{wb_id}\n")


if __name__ == '__main__':
    run()