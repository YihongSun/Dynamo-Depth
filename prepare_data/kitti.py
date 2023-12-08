import os, sys
import os.path as osp
import numpy as np
from kitti_util import generate_depth_map
from PIL import Image
from tqdm import tqdm


raw_dataset = sys.argv[1]
out_dataset = sys.argv[2]

cam_names = ['image_02', 'image_03']

downsample_height, downsample_width = 192, 640

bool_rgb = True
bool_depth = True

img_d_name = 'rgb'
depth_d_name = 'depth'

"""
<out_dataset>/
|-- 2011_09_26
|   |-- 2011_09_26_drive_0001_sync
|   |   |-- image_02
|   |   |   |-- rgb
|   |   |   |   |-- original
|   |   |   |   |-- downsample
|   |   |   |-- depth
|   |   |-- image_03
|   |   |   |-- rgb
|   |   |   |-- depth
|   |-- 2011_09_26_drive_0005_sync
|   |-- ...
|-- 2011_09_28
|   |-- ...
|-- 2011_09_29
|   |-- ...
|-- 2011_09_30
|   |-- ...
|-- 2011_10_03
|   |-- ...
"""

if not osp.exists(out_dataset):
    os.mkdir(out_dataset)

traversal_dates = [f for f in os.listdir(raw_dataset) if f.startswith("2011")]
traversal_dates.sort()

for t_date in traversal_dates:

    t_date_dir = osp.join(raw_dataset, t_date)
    t_date_out_dir = osp.join(out_dataset, t_date)
    if not osp.exists(t_date_out_dir):
        os.mkdir(t_date_out_dir)
    
    traversal_names = [f for f in os.listdir(t_date_dir) if f.startswith(t_date)]

    iterator = tqdm(traversal_names, desc=f'Processing {t_date}')
    for t_name in iterator:
        t_dir = osp.join(raw_dataset, t_date, t_name)
        t_out_dir = osp.join(out_dataset, t_date, t_name)
        if not osp.exists(t_out_dir):
            os.mkdir(t_out_dir)

        iterator.set_postfix({"traversal": t_name})

        for txt_file in [f for f in os.listdir(t_date_dir) if f.endswith('.txt')]:
            txt_src_path = osp.join(t_date_dir, txt_file)
            txt_out_path = osp.join(t_out_dir, txt_file)
            if not osp.exists(txt_out_path):
                os.system(f'ln -s {osp.realpath(txt_src_path)} {txt_out_path}')

        for cam_name in cam_names:
            
            cam_out_dir = osp.join(t_out_dir, cam_name)         
            img_out_dir = osp.join(cam_out_dir, img_d_name)
            depth_out_dir = osp.join(cam_out_dir, depth_d_name)
            org_img_out_dir = osp.join(img_out_dir, 'original')
            down_img_out_dir = osp.join(img_out_dir, 'downsample')
            
            for d in [cam_out_dir, img_out_dir, depth_out_dir, org_img_out_dir, down_img_out_dir]:
                if not osp.exists(d):
                    os.mkdir(d)
            
            img_src_dir = osp.join(t_dir, cam_name, 'data')
            depth_src_dir = osp.join(t_dir, 'velodyne_points', 'data')
            img_names = [f.split('.')[0] for f in os.listdir(img_src_dir) if f.endswith('.png')]

            if bool_rgb: # Process Image
                for img_name in img_names:
                    img_src_path = osp.join(img_src_dir, f'{img_name}.png')
                    org_out_path = osp.join(org_img_out_dir, f'{img_name}.png') 
                    down_out_path = osp.join(down_img_out_dir, f'{img_name}.jpg') 

                    if not osp.exists(org_out_path):
                        os.system(f'ln -s {osp.realpath(img_src_path)} {org_out_path}')
                    if not osp.exists(down_out_path):
                        img = Image.open(img_src_path)
                        img.resize((downsample_width, downsample_height)).save(down_out_path)
            
            if bool_depth: # Process Depth
                for img_name in img_names:
                    depth_src_path = osp.join(depth_src_dir, f'{img_name}.bin') 
                    depth_out_path = osp.join(depth_out_dir, f'{img_name}.npy') 
                    if not osp.exists(depth_src_path):
                        print(f'Depth Data {depth_src_path} Not Found - Skipped')
                        continue
                    depth_map = generate_depth_map(t_date_dir, depth_src_path, cam=int(cam_name[-1]), vel_depth=True)
                    h_ind, w_ind = np.where(depth_map>0)
                    d = depth_map[h_ind,w_ind]
                    depth_points = np.stack([h_ind, w_ind, d]).transpose((1,0))
                    np.save(depth_out_path, depth_points)

    
    
