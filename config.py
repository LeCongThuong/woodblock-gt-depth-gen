from dataclasses import dataclass


@dataclass
class args:
    woodblock_path: str = 'notebooks/woodblock_xyz/whole_02801_mk30_xyz.stl'
    surface_path: str = "notebooks/surface_xyz/surface_02801_mk30_xyz.stl"
    scan_img: str = "notebooks/scan_imgs/02801_mk30.jpg"
    pitch_map: str = "notebooks/pitch_map/pitch_map_02801_mk30.json"
    sina_nom_anno: str = "notebooks/sina_nom_annos/02801_mk30.json"
    aligned_matrix: str = "notebooks/woodblock_inverted_matrix/whole_02801_mk30_xyz.npy"
    character_dest: str = "notebooks/character_3d_aligned"
