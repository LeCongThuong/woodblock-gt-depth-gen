from pathlib import Path
import os
import argparse
from tqdm import tqdm


def quality_assurance_check(wb_id_dir_path):
    num_files_status = check_number_of_files(wb_id_dir_path)
    name_files_status = check_name_of_all_files(wb_id_dir_path)
    size_files_status = check_size_of_all_files(wb_id_dir_path)
    if size_files_status and num_files_status and name_files_status:
        return True
    else:
        return False


def check_number_of_files(wb_id_dir_path, num_file=5):
    file_path_list = list(Path(wb_id_dir_path).glob("*"))
    is_ok = True if len(file_path_list) == num_file else False
    return is_ok


def check_name_of_all_files(wb_id_dir_path):
    wb_id = Path(wb_id_dir_path).stem
    print(wb_id)
    file_path_list = list(Path(wb_id_dir_path).glob("*"))
    has_sf = False
    has_fl = False
    has_sina_nom = False
    has_print_img = False
    for file_path in file_path_list:
        file_name = file_path.name
        if wb_id not in file_name:
            return False
        if "surface" in file_name:
            has_sf = True
        if "floor" in file_name:
            has_fl = True
        if "json" in file_name:
            has_sina_nom = True
        if "jpg" in file_name:
            has_print_img = True
    if has_sf and has_fl and has_sina_nom and has_print_img:
        return True
    else:
        return False


def check_size_of_all_files(wb_id_dir_path, threshold=50):
    file_path_list = list(Path(wb_id_dir_path).glob("*"))
    less_thresh = 0
    more_thresh = 0
    for file_path in file_path_list:
        file_size_in_mb = os.path.getsize(str(file_path)) / (1024*1024)
        if file_size_in_mb < threshold:
            less_thresh += 1
        else:
            more_thresh += 1
    if less_thresh == 4 and more_thresh == 1:
        return True
    else:
        return False


def parse_aug():
    parser = argparse.ArgumentParser(prog='QA data')
    parser.add_argument('-data', '--data_dir', help='path to directory listing id of all the woodblocks')
    parser.add_argument('-output', '--output', help='path to the file listing all the error woodblock id')
    args = parser.parse_args()
    return args


def main():
    args = parse_aug()
    woodblock_id_dir_list = list(Path(args.data_dir).glob("*"))
    with open(args.output, 'w') as f:
        for woodblock_id_dir in tqdm(woodblock_id_dir_list):
            woodblock_id = woodblock_id_dir.stem
            qa_status = quality_assurance_check(woodblock_id_dir)
            if not qa_status:
                f.write(f'{woodblock_id}\n')


if __name__ == '__main__':
    main()
