import os, cv2
import skimage.transform
import numpy as np
import PIL.Image as pil
from .base_dataset import BaseDataset

class KITTIDataset(BaseDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)


        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

        self.categories = {
            0 :' unlabeled',
            1 :' ego vehicle',
            2 :' rectification border',
            3 :' out of roi',
            4 :' static',
            5 :' dynamic',
            6 :' ground',
            7 :' road',
            8 :' sidewalk',
            9 :' parking',
            10 :' rail track',
            11 :' building',
            12 :' wall',
            13 :' fence',
            14 :' guard rail',
            15 :' bridge',
            16 :' tunnel',
            17 :' pole',
            18 :' polegroup',
            19 :' traffic light',
            20 :' traffic sign',
            21 :' vegetation',
            22 :' terrain',
            23 :' sky',
            24 :' person',
            25 :' rider',
            26 :' car',
            27 :' truck',
            28 :' bus',
            29 :' caravan',
            30 :' trailer',
            31 :' train',
            32 :' motorcycle',
            33 :' bicycle',
            -1 :' license plate',
        }

    def get_timestep(self, folder, frame_index, offset):
        return 1    # consistent timesteps in this dataset

    def get_intrinsic(self, folder):
        return self.K

    def get_gt_dim(self, folder, frame_index, side):

        with open(os.path.join(self.data_path, folder, 'calib_cam_to_cam.txt'), 'r') as f:
            _, width, height = [l for l in f.read().splitlines() if f'S_rect_0{self.side_map[side]}' in l][0].split()
        return int(float(height)), int(float(width))
    
    def get_img_path(self, folder, frame_index, side):
        cam_name = f'image_0{self.side_map[side]}' 
        return os.path.join(self.data_path, folder, cam_name, 'rgb', self.img_type, f'{frame_index:010}{self.img_ext}')

    def get_color(self, folder, frame_index, side, do_flip):

        if frame_index == -1:
            frame_index = 0

        color = self.loader(self.get_img_path(folder, frame_index, side))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
    
    def get_depth(self, folder, frame_index, side, do_flip):
        
        if frame_index == -1:
            frame_index = 0

        cam_name = f'image_0{self.side_map[side]}' 
        depth_path = os.path.join(self.data_path, folder, cam_name, 'depth', f'{frame_index:010}.npy')

        depth = np.load(depth_path)

        if do_flip:
            depth[:,1] = self.full_res_shape[0] - depth[:,1]

        out_bound = depth[:,0] >= self.full_res_shape[1]
        depth[:,0][out_bound] = self.full_res_shape[1] - 1

        out_bound = depth[:,1] >= self.full_res_shape[0]
        depth[:,1][out_bound] = self.full_res_shape[0] - 1

        return depth

    def get_mask(self, folder, frame_index, side, do_flip):
        
        if frame_index == -1:
            frame_index = 0

        cam_name = f'image_0{self.side_map[side]}' 

        mot_path = os.path.join(self.data_path, folder, cam_name, 'mask', f'{frame_index:010}_mot.npy')
        sem_path = os.path.join(self.data_path, folder, cam_name, 'mask', f'{frame_index:010}_sem.npy')

        if not os.path.exists(sem_path):    # in case mask annotation is not found
            return np.zeros(self.full_res_shape[::-1]), np.zeros(self.full_res_shape[::-1])

        sem_mask = np.load(sem_path)
        mot_mask = np.load(mot_path)

        if mot_mask.shape[0] != self.full_res_shape[1] or mot_mask.shape[1] != self.full_res_shape[0]:
            sem_mask = cv2.resize(sem_mask, self.full_res_shape, cv2.INTER_NEAREST)
            mot_mask = cv2.resize(mot_mask, self.full_res_shape, cv2.INTER_NEAREST)

        return sem_mask, mot_mask


def generate_depth_map(calib_dir, velo_filename, cam=2, vel_depth=False):
    """Generate a depth map from velodyne data
    """
    # load calibration files
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # get image shape
    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_filename)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    return velo_pts_im

class KITTIDatasetOld(BaseDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDatasetOld, self).__init__(*args, **kwargs)

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
    
    def get_intrinsic(self, folder=None):
        return self.K


class KITTIRAWDataset(KITTIDatasetOld):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}_192{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class KITTIOdomDataset(KITTIDatasetOld):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path


class KITTIDepthDataset(KITTIDatasetOld):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
