from dataclasses import dataclass


@dataclass
class args:
    woodblock_path: str = 'notebooks/woodblock_xyz/whole_02801_mk30_xyz.stl'
    character_surface: str = "notebooks/test_moc_ban_ok/surface_02801_mk30.stl"
    floor: str = "notebooks/test_moc_ban_ok/floor_02801_mk30.stl"
    scan_img: str = "notebooks/scan_imgs/02801_mk30.jpg"
    pitch_map: str = "notebooks/pitch_map/pitch_map_02801_mk30.json"
    raw_pitch_map: str = "notebooks/raw_pitch_map/pitch_map_02801_mk30.json"
    sina_nom_anno: str = "notebooks/sina_nom_annos/02801_mk30.json"
    inverted_matrix: str = "notebooks/woodblock_inverted_matrix/whole_02801_mk30_xyz.npy"
    character_dest: str = "notebooks/character_3d_aligned"
