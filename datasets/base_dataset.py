import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class BaseDataset(data.Dataset):
    """Superclass for dataloaders"""

    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 cam_name,
                 img_type,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg',
                 load_depth=False,
                 load_mask=False,
                 path=False):
        super(BaseDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width

        self.cam_name = cam_name
        self.img_type = img_type

        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS
        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            self.jitter = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1
            self.jitter = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        self.aug_freq = 0.5

        self.give_path = path

        self.load_mask = load_mask
        self.load_depth = load_depth
        self.max_lidar_num = 25000  #used to pad for batching

        self.img_resize = transforms.Resize((self.height, self.width), interpolation=transforms.InterpolationMode.BICUBIC)
        

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """

        for im in self.frame_idxs:
            f = self.to_tensor(inputs[("color", im, 0)])
            inputs[("color", im, 0)] = f
            inputs[("color_aug", im, 0)] = color_aug(f)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        frame_index = int(line[1])  #hacky might give error for kitti but not sure why rn

        if len(line) == 3:
            side = line[2]
        else:
            side = 'l' # assume left - image_02 if not given

        for i in self.frame_idxs:
            if i == "s":
                raise Exception('Current functionality does not handle stereo inputs')
            
            inputs[("color", i, 0)] = self.get_color(folder, frame_index + i, side, do_flip)
            inputs[("ts", i)] = self.get_timestep(folder, frame_index, i)
            
            load_height, load_width = inputs[("color", i, 0)].size[1], inputs[("color", i, 0)].size[0]
            
            if load_height != self.height or load_width != self.width:
                inputs[("color", i, 0)] = self.img_resize(inputs[("color", i, 0)])
            
            gt_height, gt_width = self.get_gt_dim(folder, frame_index + i, side)

            inputs["gt_dim"] = torch.tensor([gt_height, gt_width]).type(torch.int)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.get_intrinsic(folder).copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if self.is_train and random.random() < self.aug_freq:
            color_aug = self.jitter
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)   # (N, 3)

            depth = torch.from_numpy(depth_gt.astype(np.float32))
            valid = torch.ones(depth_gt.shape[0])

            # Pad for batching with different N
            depth = torch.cat((depth, torch.zeros(self.max_lidar_num-depth.shape[0], 3)))
            valid = torch.cat((valid, torch.zeros(self.max_lidar_num-valid.shape[0])))
                
            inputs["depth_gt"] = depth      # (N, 3)
            inputs["depth_valid"] = valid   # (N)
        
        if self.load_mask:
            sem_mask, mot_mask = self.get_mask(folder, frame_index, side, do_flip)

            inputs["sem_mask"] = torch.from_numpy(sem_mask).type(torch.uint8)   # self.full_res_shape[::-1]
            inputs["mot_mask"] = torch.from_numpy(mot_mask).type(torch.uint8)   # self.full_res_shape[::-1]

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)
        
        if self.give_path:
            inputs['paths'] = line
        
        inputs['index'] = index
            
        return inputs
    
    def get_img_path(self, folder, frame_index, side):
        raise NotImplementedError

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def get_mask(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
    
    def get_intrinsic(self, folder):
        raise NotImplementedError
