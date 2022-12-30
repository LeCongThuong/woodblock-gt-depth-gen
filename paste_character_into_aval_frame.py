import json
import os
from parsing import read_json_file, parse_bboxes_list_from_sino_nom_anno_file
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


def crop_2d_img(img, pts, tgt_size):
    pts = np.array(pts, dtype=np.int32)
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    scale_ratio = tgt_size / max(w, h)
    croped = img[y:y+h, x:x+w].copy()
    croped = cv2.resize(croped, (round(w*scale_ratio), round(h*scale_ratio)), interpolation = cv2.INTER_AREA)
    w, h = round(w*scale_ratio), round(h*scale_ratio)
    bg_center = (tgt_size//2, tgt_size//2)
    bg_img = np.full((tgt_size, tgt_size, 3), 255)
    t_l_x, t_l_y = bg_center[0] - w // 2, bg_center[1] - h // 2
    bg_img[t_l_y: t_l_y + h, t_l_x: t_l_x + w, :] = croped
    return bg_img


def run_a_file(print_img_path, sino_nom_anno_path, character_dest_path, tgt_size=256):
    bboxes_2d_list = parse_bboxes_list_from_sino_nom_anno_file(sino_nom_anno_path)
    print_img = cv2.imread(print_img_path)
    for s_index, bbox in enumerate(bboxes_2d_list):
        character_img = crop_2d_img(print_img, bbox, tgt_size)
        index_character_dest_path = os.path.join(character_dest_path, f"{s_index}.png")
        cv2.imwrite(index_character_dest_path, character_img)


def run_files(parent_out_dir, parent_raw_dir):
    wb_id_list = [wb_path.name for wb_path in list(Path(parent_raw_dir).glob("*"))]
    for wb_id in tqdm(wb_id_list):
        print_img_path = os.path.join(parent_raw_dir, wb_id, f"nb_aligned_{wb_id}.png")
        sino_nom_anno_path = os.path.join(parent_raw_dir, wb_id, f"nb_aligned_{wb_id}.json")
        character_dest_path = os.path.join(parent_out_dir, wb_id, "character_xyz", "normalized_print")
        Path(character_dest_path).mkdir(exist_ok=True, parents=True)
        run_a_file(print_img_path, sino_nom_anno_path, character_dest_path)


if __name__ == '__main__':
    parent_out_dir = "data/output"
    parent_raw_dir = "data/raw"
    run_files(parent_out_dir, parent_raw_dir)