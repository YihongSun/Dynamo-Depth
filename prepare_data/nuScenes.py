import os, sys, json
import numpy as np
import os.path as osp
import cv2
from pyquaternion import Quaternion
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes  #pip install nuscenes-devkit
from nuscenes.utils.data_io import load_bin_file
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix



def join_dir(*items):
    d = osp.join(*items)
    os.makedirs(d, exist_ok=True)
    return d

def get_linked_list(first_item, table_name):
    arr = [first_item]
    while arr[-1]['next'] is not '':
        new_item = nusc.get(table_name, arr[-1]['next'])
        assert new_item['prev'] == arr[-1]['token']
        arr.append(new_item)
    return arr

def get_depth_points(lidar_token, cam_token):
    if type(lidar_token) is not list:
        lidar_token = [lidar_token]
    
    depth_points = []
    for lidar_t in lidar_token:
        depth_coord, depth_val, im = nusc.explorer.map_pointcloud_to_image(lidar_t, cam_token, render_intensity=False, show_lidarseg=False, filter_lidarseg_labels=None, lidarseg_preds_bin_path=None, show_panoptic=False)

        d_points = depth_coord.T
        d_points[:,2] = depth_val

        depth_points.append(d_points)

    return np.vstack(depth_points)

def get_intersect_fraction(points, corners):
    """ find the percentage of the input points that are within the 3D box defined by the input corners
    """
    if points.shape[0] == 0:
        return 0

    p1, p2, p4, p5 = corners[0], corners[1], corners[3], corners[4]
    i_vec, j_vec, k_vec, v_vec = p2 - p1, p4 - p1, p5 - p1, points - p1
    vi, vj, vk = v_vec @ i_vec.T, v_vec @ j_vec.T, v_vec @ k_vec.T
    ii, jj, kk = i_vec @ i_vec.T, j_vec @ j_vec.T, k_vec @ k_vec.T

    return ((0 < vi) & (vi < ii) & (0 < vj) & (vj < jj) & (0 < vk) & (vk < kk)).mean()


if __name__ == '__main__':
    data_root = sys.argv[1]
    data_version = 'v1.0-trainval'
    cam_channel = 'CAM_FRONT'
    lidar_channel = 'LIDAR_TOP'
    cam_name = cam_channel[4:]  # ignore 'CAM_'
    downsample_factor = 3.125
    nsweeps = 1

    img_d_name = 'rgb'
    depth_d_name = 'depth'
    mask_d_name = 'mask'

    nusc = NuScenes(version=data_version, dataroot=data_root, verbose=True)

    cat2idx = {c['name'] : c['index'] for c in nusc.category}

    moving_attr_tokens = {a['token'] for a in nusc.attribute if 'moving' in a['name']}
    movable_cat_ind = {c['index'] for c in nusc.category if 'animal' in c['name'] or 'human' in c['name'] or 'vehicle' in c['name']}
    movable_cat_ind.remove(31)   # removing vehicle.ego : since camera is mounted, ego car is not movable

    """
    <dataset_dir>/
    |-- v1.0-trainval/
    |-- nuScenes-panoptic-v1.0-all/
    |-- maps/
    |-- samples/
    |-- sweeps/
    |-- scenes/
    |   |-- scene-...
    |   |   |-- FRONT
    |   |   |   |-- rgb
    |   |   |   |   |-- original
    |   |   |   |   |-- downsample
    |   |   |   |-- depth
    |   |   |   |-- mask
    |   |   |-- FRONT_LEFT
    |   |   |   |-- ...
    |   |-- scene-...
    |   |   |-- ...
    |   |-- ...
    """
    iterator = tqdm(enumerate(nusc.scene), desc='Processing scenes', total=len(nusc.scene))
    for s_idx, sc in iterator:
        scene_token = sc['token']
        scene_name = sc['name']
        log_token = sc['log_token']
        nbr_samples = sc['nbr_samples']
        first_sample_token = sc['first_sample_token']
        last_sample_token = sc['last_sample_token']

        first_sample = nusc.get('sample', first_sample_token)
        samples = get_linked_list(first_sample, 'sample')       # used for consistency only

        first_cam = nusc.get('sample_data', first_sample['data'][cam_channel])
        cams = get_linked_list(first_cam, 'sample_data')
        sample_cams = [c for c in cams if c['is_key_frame']]

        first_lidar = nusc.get('sample_data', first_sample['data'][lidar_channel])
        unmapped_lidars = get_linked_list(first_lidar, 'sample_data')
        lidars = [
            unmapped_lidars[i] for i in np.array([[abs(l['timestamp']-cam['timestamp']) for l in unmapped_lidars] for cam in cams]).argmin(1)
        ]
        for ii, cam in enumerate(cams): # override for already associated depth
            if cam['is_key_frame']:
                lidars[ii] = nusc.get('sample_data', nusc.get('sample', cam['sample_token'])['data']['LIDAR_TOP'])

        assert len(samples) == len(sample_cams) == nbr_samples, 'Number of samples should be consistent.'
        assert np.all([s['token'] == c['sample_token'] for s, c in zip(samples, sample_cams)]), 'Order of samples and sample_cams should be consistent'
        assert samples[-1]['token'] == last_sample_token, 'last_sample_token should be consistent.'

        # declare and create directories
        org_rgb_d = join_dir(data_root, 'scenes', scene_name, cam_name, img_d_name, 'original')
        dwn_rgb_d = join_dir(data_root, 'scenes', scene_name, cam_name, img_d_name, 'downsample')
        depth_d = join_dir(data_root, 'scenes', scene_name, cam_name, depth_d_name)
        mask_d = join_dir(data_root, 'scenes', scene_name, cam_name, mask_d_name)
        cam_path = osp.join(data_root, 'scenes', scene_name, cam_name, img_d_name, 'cam.json')

        poses = []
        for ii, cam in enumerate(cams):
            cam_token = cam['token']
            sample_token = cam['sample_token']
            is_key_frame = cam['is_key_frame']
            org_height, org_width = cam['height'], cam['width']

            org_path =      osp.join(org_rgb_d, '{:06}.jpg'.format(ii))
            dwn_path =      osp.join(dwn_rgb_d, '{:06}.jpg'.format(ii))
            depth_path =    osp.join(depth_d, '{:06}.npy'.format(ii))
            mask_path =     osp.join(mask_d, '{:06}.npz'.format(ii))

            ## Process Image
            fpath = osp.join(data_root, cam['filename'])
            if not osp.exists(org_path):
                os.system(f'ln -s {fpath} {org_path}')
            if not osp.exists(dwn_path):
                dwn_height, dwn_width = int(org_height / downsample_factor), int(org_width / downsample_factor)
                dwn_rgb = cv2.resize(cv2.imread(fpath), (dwn_width, dwn_height), interpolation=cv2.INTER_AREA)
                cv2.imwrite(dwn_path, dwn_rgb)

            lidar = lidars[ii]
            pcl_path = osp.join(data_root, lidar['filename'])
            pc = LidarPointCloud.from_file(pcl_path)

            # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
            # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
            cs_record = nusc.get('calibrated_sensor', lidar['calibrated_sensor_token'])
            pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
            pc.translate(np.array(cs_record['translation']))

            # Second step: transform from ego to the global frame.
            poserecord = nusc.get('ego_pose', lidar['ego_pose_token'])
            pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
            pc.translate(np.array(poserecord['translation']))

            global_points = pc.points.T[:,:3].copy() # used to rectify with 3d bbox

            # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
            poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
            pc.translate(-np.array(poserecord['translation']))
            pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

            # Fourth step: transform from ego into the camera.
            cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
            pc.translate(-np.array(cs_record['translation']))
            pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

            # Fifth step: actually take a "picture" of the point cloud.
            # Grab the depths (camera frame z axis points away from the camera).
            depths = pc.points[2, :]
            points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

            min_dist = 1.0
            mask = np.ones(depths.shape[0], dtype=bool)
            mask = np.logical_and(mask, depths > min_dist)
            mask = np.logical_and(mask, points[0, :] > 1)
            mask = np.logical_and(mask, points[0, :] < org_width - 1)
            mask = np.logical_and(mask, points[1, :] > 1)
            mask = np.logical_and(mask, points[1, :] < org_height - 1)

            depth_coord = points[:,mask].T[:,:2]
            depth_vals = depths[mask]
            depth_points = np.hstack((depth_coord, depth_vals[:,np.newaxis]))
            lidar_points = global_points[mask]

            ## Process Depth
            if not osp.exists(depth_path):
                if is_key_frame:
                    assert lidar['token'] == nusc.get('sample', sample_token)['data']['LIDAR_TOP']
                np.save(depth_path, depth_points)
            else:
                assert np.linalg.norm(np.load(depth_path) - depth_points) == 0

            ## Process Mask (LiDAR)
            if is_key_frame and not osp.exists(mask_path):
                panoptic_labels_filename = osp.join(data_root, nusc.get('panoptic', lidar['token'])['filename'])
                panoptic_labels = load_bin_file(panoptic_labels_filename, type='panoptic')
                panoptic = panoptic_labels[mask]

                token2cat = {b.token : cat2idx[b.name]  for b in nusc.get_boxes(lidar['token'])}
                token2box = {b.token : b.corners().T  for b in nusc.get_boxes(lidar['token'])}
                token2attr = {nusc.get('sample_annotation', ann)['token'] : nusc.get('sample_annotation', ann)['attribute_tokens'] for ann in nusc.get('sample', sample_token)['anns']}

                motion_label = np.ones_like(panoptic) * 3 # unlabeled - couldn't find a match
                panoptic2ann = dict()
                for unique_label in np.unique(panoptic):
                    cat = unique_label // 1000
                    panoptic_mask = panoptic==unique_label

                    if cat not in movable_cat_ind:
                        motion_label[panoptic_mask] = 0
                    else:
                        #unique_label contains LiDAR points on a movable object - find best fitted box
                        btoken, bfit = None, 0
                        for ann_token in token2cat:
                            if token2cat[ann_token] != cat:
                                continue
                            frac = get_intersect_fraction(lidar_points[panoptic_mask], token2box[ann_token])
                            if frac > bfit:
                                bfit = frac
                                btoken = ann_token
                        if btoken is None:
                            motion_label[panoptic_mask] = 3 # unlabeled - couldn't find a match
                        elif np.any([at in moving_attr_tokens for at in token2attr[btoken]]):
                            motion_label[panoptic_mask] = 1 # in motion
                        else:
                            motion_label[panoptic_mask] = 2 # static movable
                        
                        panoptic2ann[unique_label] = {'token' : btoken, 'fit' : bfit}
                np.savez_compressed(mask_path, panoptic_label=panoptic, panoptic2ann=panoptic2ann, motion_label=motion_label.astype(np.uint8))

            ## Process Camera Intrinsics
            if not osp.exists(cam_path):  # should only run in the first cam
                calibrated_sensor = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
                translation = calibrated_sensor['translation']
                rotation = calibrated_sensor['rotation']
                camera_intrinsic = np.array(calibrated_sensor['camera_intrinsic'])
                camera_intrinsic[0] /= org_width
                camera_intrinsic[1] /= org_height
                int_mat = camera_intrinsic.tolist()

                with open(cam_path, 'w') as fh: 
                    json.dump({ 'camera_intrinsic' : calibrated_sensor['camera_intrinsic'], 
                                'translation' : calibrated_sensor['translation'], 
                                'rotation' : calibrated_sensor['rotation'], 
                                'dim' : [org_height, org_width], 
                                'intrinsic_mat': int_mat,
                                }, fh)

            calib_data = nusc.get("calibrated_sensor", cam["calibrated_sensor_token"])
            egopose_data = nusc.get('ego_pose',  cam["ego_pose_token"])
            car_to_velo = transform_matrix(calib_data["translation"], Quaternion(calib_data["rotation"]))
            pose_car = transform_matrix(egopose_data["translation"], Quaternion(egopose_data["rotation"]))
            # pose = np.dot(pose_car, car_to_velo)
            poses.append(' '.join([str(x) for x in pose_car.flatten()]))

        assert len(os.listdir(org_rgb_d)) == len(poses)
        with open(osp.join(data_root, 'scenes', scene_name, cam_name, 'odometry.txt'), 'w') as fh:
            for line in poses:
                fh.write(line + '\n')
        
        time_steps = np.array([np.rint((c2['timestamp'] - c1['timestamp']) / 1000) for c1,c2 in zip(cams[:-1],cams[1:])]).astype(np.uint8).tolist() # in milisec
        with open(osp.join(data_root, 'scenes', scene_name, cam_name, img_d_name, 'ts.json'), 'w') as fh: 
            json.dump(time_steps, fh)

        iterator.set_postfix({"scene": scene_name})

 