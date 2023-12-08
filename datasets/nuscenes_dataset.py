import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import pickle, json, cv2
import torch, torchvision

from .base_dataset import BaseDataset

class nuScenesDataset(BaseDataset):
    """Superclass for different types of Waymo dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(nuScenesDataset, self).__init__(*args, **kwargs)

        self.K = dict()
        self.get_all_intrinsic()

        self.full_res_shape = (1600, 900)
        self.median_ts = 100.0              # process_datasets/process_nuscenes_timestep.py

    def get_all_intrinsic(self):
        for file in self.filenames:
            folder = file.split()[0]
            if folder not in self.K:
                self.K[folder] = np.eye(4, dtype=np.float32)

                cam_path = os.path.join(self.data_path, folder, self.cam_name, 'rgb', 'cam.json')

                with open(cam_path, 'r') as fh:
                    self.K[folder][:3, :3] = np.array(json.load(fh)['intrinsic_mat'])
    
    def get_timestep(self, folder, frame_index, offset):
        ''' Obtain the amount of time that passed between frame_index and the offset index
        '''
        ts_path = os.path.join(self.data_path, folder, self.cam_name, 'rgb', 'ts.json')
        with open(ts_path, 'r') as fh:
            timesteps = json.load(fh)
        low, high = min(frame_index, frame_index+offset), max(frame_index, frame_index+offset)
        return np.sum(timesteps[low:high]) / self.median_ts  # relative_ts = absolute duration / median

    def get_intrinsic(self, folder):
        return self.K[folder]
    
    def get_gt_dim(self, folder, frame_index, side):
        return self.full_res_shape[1], self.full_res_shape[0]
    
    def get_img_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        return os.path.join(self.data_path, folder, self.cam_name, 'rgb', self.img_type, f_str)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_img_path(folder, frame_index, side))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color
    
    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:06d}{}".format(frame_index, '.npy')
        depth_path = os.path.join(self.data_path, folder, self.cam_name, 'depth', f_str)

        depth = np.load(depth_path)

        if do_flip:
            depth[:,0] = self.full_res_shape[0] - depth[:,0]
        
        depth = np.concatenate((depth[:,1:2], depth[:,0:1], depth[:,2:3]), axis=1)    # (N, 3) -> [h_i, w_i, z_i]

        return depth

    def get_mask(self, folder, frame_index, side, do_flip):
        f_str_mask = "{:06d}{}".format(frame_index, '.npz')
        mask_path = os.path.join(self.data_path, folder, self.cam_name, 'mask', f_str_mask)

        if not os.path.exists(mask_path):    # in case mask annotation is not found
            return np.zeros(self.full_res_shape[::-1]), np.ones(self.full_res_shape[::-1]) * 3

        mask_ann = np.load(mask_path)
        motion_seg = mask_ann['motion_label']

        depth_points = self.get_depth(folder, frame_index, side, do_flip=False)
        lidar_coord = depth_points[:,:2]
        
        scale = 5
        org_width, org_height = self.full_res_shape
        dwn_width, dwn_height = org_width // scale, org_height // scale

        lidar_coord = (torch.tensor(lidar_coord)/scale).long()
        lidar_coord[lidar_coord < 0] = 0
        lidar_coord[:,0][lidar_coord[:,0] >= dwn_height] = dwn_height - 1
        lidar_coord[:,1][lidar_coord[:,1] >= dwn_width] = dwn_width - 1

        mot_seg = torch.ones(dwn_height, dwn_width) * 3
        mot_seg[lidar_coord[:,0], lidar_coord[:,1]] = torch.tensor(motion_seg).float()
        mot_seg = torchvision.transforms.Resize((org_height, org_width), interpolation=torchvision.transforms.InterpolationMode.NEAREST)(mot_seg.unsqueeze(0))[0].numpy()
        
        return np.ones(self.full_res_shape[::-1]), mot_seg

    
