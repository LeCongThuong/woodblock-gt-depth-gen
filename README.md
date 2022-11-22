The goal of this repository is to generate depth maps images from 3D woodblock models.
 
![alt text](assets/depth_map_generation.png)

The depth map ground-truth image generation contains 5 main steps:
* Step 1: 
  * Get the depth map of 3D model after the pitching and rotation transform process.
  * Command: python3 z_transform.py
    * --id_path: path to the file listing all id of woodblocks (wb) (each id on each line).
    * --raw    : the parent directory contains the raw wb data directories.  
    * --dest   : the interim directory contains by-product result such as z_depth images, v.v.
    * --error  : path to the file listing error id of woodblocks in process.
    * --mirror : used if the woodblock id down-side
* Step 2:
  * Use the VIA tool to get border points of woodblock in depth map generated in step 1.
* Step 3:
  * Get the aligned depth map of 3D model and 3D aligned models.
  * Command: python3 xyz_transform.py
    * --id_path: path to the file listing all id of woodblocks (wb) (each id on each line).
    * --raw    : the parent directory contains the raw wb data directories. 
    * --mirror : path to file listing all wb ids that need to mirror in step 1.
    * --interim: path to directory contains by-product result such as z_depth, surface_xyz, border points, v.v..
    * --output : path to the destination directory to save aligned model, aligned depth .
    * --error  : path to the file listing error id of woodblocks in process.
* Step 4:
  * Use the VIA tool to get mapping points between an aligned depth image and scan image.
* Step 5:
  * Get characters of an wb.
  * Command: python3 cropping.py 
  * --id_path: path to the file listing all id of woodblocks (wb) (each id on each line).
    * --raw    : the parent directory contains the raw wb data directories.
    * --interim: path to directory contains by-product result such as z_depth, surface_xyz, border points, v.v..
    * --output : path to the destination directory to save aligned model, aligned depth .
    * --error  : path to the file listing error id of woodblocks in process.
* Example:
  * Do batch z_transform (Step 1)
    * python3 z_transform.py -id_path "data/wb_id.txt" -raw "data/raw" -dest "data/interim" -error "data/z_trans_error.txt"
  * Do batch xyz_transform (Step 3)
    * python3 xyz_transform.py -id_path "data/wb_id.txt" -raw "data/raw" -mirror "data/mirror_wb_id.txt" -interim "data/interim" -output "data/output" -error "data/xyz_trans_error.txt"
  * Do crop transform (Step 5)
    * python3 cropping.py -id_path "data/wb_id.txt" -raw "data/raw" -interim "data/interim" -output "data/output" -error "data/cropping_error.txt"