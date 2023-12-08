import os, sys, json, cv2, time, pickle
import os.path as osp
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils, box_utils, camera_segmentation_utils


def calibrate(img, intrinsic, dim):
    f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3 = intrinsic

    cameraMatrix = np.eye(3)
    cameraMatrix[0,0] = f_u
    cameraMatrix[0,2] = c_u
    cameraMatrix[1,1] = f_v
    cameraMatrix[1,2] = c_v
    
    distCoeffs = np.array([k1, k2, p1, p2, k3])

    out_img = cv2.undistort(img, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)

    height, width = dim
    cameraMatrix[0] /= width
    cameraMatrix[1] /= height

    return out_img, cameraMatrix.tolist()

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

def get_instance_masks(semantic_label, instance_label, labels=None):
    """ return individual masks based on the input semantic and instance label maps
    """
    if labels == None:
        labels = list(range(1, semantic_label.max()))

    ind_masks, ind_labels = list(), list()

    for c in labels:
        cls_instance_label = (instance_label+1) * (semantic_label == c).astype(int) 

        for i in range(1, cls_instance_label.max()+1):
            ind_masks.append((cls_instance_label == i).astype(int))
            ind_labels.append(c)

    return ind_masks, ind_labels


if __name__ == "__main__":
    # Hyperparameters
    bool_cam = True        
    bool_depth = True
    bool_MotMask = True
    cam_names = ['FRONT']  # ['FRONT', 'FRONT_LEFT', 'SIDE_LEFT', 'FRONT_RIGHT', 'SIDE_RIGHT']
    downsample_factor = 4
    img_d_name = 'rgb'
    depth_d_name = 'depth'
    mask_d_name = 'mask'

    org_height, org_width = 1280, 1920  # fixed
    moveable_categories_mask = {
        2 : 'car',
        3 : 'truck',
        4 : 'bus',
        5 : 'other_vehicle',
        6 : 'bicycle',
        7 : 'motorcycle',
        8 : 'trailer',
        9 : 'pedestrian',
        10 : 'bicyclist',
        11 : 'motorcyclist',
        12 : 'bird',
        13 : 'ground_animal',
        16 : 'pedestrian_object',
        27 : 'dynamic',
    }
    frame = open_dataset.Frame()    # Used to load each frame

    RECORD_DIR = sys.argv[1]
    DATASET_DIR = sys.argv[2]
    if not osp.exists(DATASET_DIR):
        print('Input dataset_dir is not found, will be created.')
        os.makedirs(DATASET_DIR)
    
    print('\n##############')
    print(f'Record Directory: {RECORD_DIR}')
    print(f'Output Directory: {DATASET_DIR}')
    print('##############\n')

    for data_type in ['val', 'train']:
        dataset_dir = osp.join(DATASET_DIR, data_type)
        record_dir = osp.join(RECORD_DIR, data_type)
        if not osp.exists(record_dir):
            raise Exception(f'Directory not found: {record_dir}')

        # Retrieving all segments from tfrecords/
        traversals = [(osp.join(record_dir, f), f[:f.index('_with')]) for f in os.listdir(record_dir) if f.endswith(".tfrecord")]
        traversals.sort(key=lambda x: x[1])

        if data_type == 'train' and len(traversals) != 798:
            raise Warning(f'Number of .tfrecord files in {record_dir} is {len(traversals)} (!= 798).')
        if data_type == 'val' and len(traversals) != 202:
            raise Warning(f'Number of .tfrecord files in {record_dir} is {len(traversals)} (!= 202).')
        
        print('\n##############')
        print(f'Processing {record_dir} | # Traversals = {len(traversals)}...')
        print('##############\n')

        iterator = tqdm(enumerate(traversals), desc='Processing Waymo Traversals', total=len(traversals))
        for index, (traversal_path, traversal_name) in iterator:
            
            # ========== Handling directories ========== #

            """
            <dataset_dir>/
            |-- segment-...
            |   |-- FRONT
            |   |   |-- rgb
            |   |   |-- |-- original
            |   |   |-- |-- downsample
            |   |   |-- depth
            |   |   |-- mask
            |   |-- FRONT_LEFT
            |   |   |-- ...
            |-- segment-...
            |   |-- ...
            |-- ...
            """

            traversal_dir = osp.join(dataset_dir, traversal_name)
            for cam_name in cam_names:
                cam_dir = osp.join(traversal_dir, cam_name)

                img_dir = osp.join(cam_dir, img_d_name)
                img_org_dir = osp.join(img_dir, 'original')
                img_dow_dir = osp.join(img_dir, 'downsample')
                lidar_dir = osp.join(cam_dir, depth_d_name)
                mask_dir = osp.join(cam_dir, mask_d_name)

                for d in [traversal_dir, cam_dir, img_dir, lidar_dir, mask_dir, img_org_dir, img_dow_dir]:
                    os.makedirs(d, exist_ok=True)
            
            num_frames = 0
            dataset = tf.data.TFRecordDataset(traversal_path, compression_type='')
            poses = {cam_name : list() for cam_name in cam_names}

            # ========== Go through each frame ========== #

            for data in dataset:               # Going through each frame
                frame.ParseFromString(bytearray(data.numpy())) 
            
                # ========== RGB scans ========== #
                cam_images =    {open_dataset.CameraName.Name.Name(img.name) : img for img in frame.images}
                cam_infos =     {open_dataset.CameraName.Name.Name(cal.name) : (cal, cal.name) for cal in frame.context.camera_calibrations}

                # ========== Segmentation Masks ========== #
                cam_masks =     {open_dataset.CameraName.Name.Name(img.name) : img.camera_segmentation_label for img in frame.images}
                bool_has_mask = bool(frame.images[0].camera_segmentation_label.panoptic_label) and bool_MotMask

                if bool_depth or bool_has_mask:     # Used by both depth loading and instance mask loading
                    # ========== LiDAR scans ========== #
                    range_images, camera_projections, _, range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(frame)
                    points, cp_points = frame_utils.convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose)
                    points_all = np.concatenate(points, axis=0)                         # 3d points in vehicle frame.
                    cp_points_all = np.concatenate(cp_points, axis=0)                   # camera projection corresponding to each point.
                    points_all_tensor = tf.norm(points_all, axis=-1, keepdims=True)     # The distance between lidar points and vehicle frame origin.
                    cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)


                # ========== Go through each camera view ========== #

                for cam_name in cam_names:

                    cam_img = cam_images[cam_name]
                    cam_cal, cam_code = cam_infos[cam_name]
                    
                    if bool_cam:
                        # Calibrate images
                        rgb_distorted = cv2.cvtColor(tf.image.decode_jpeg(cam_img.image).numpy(), cv2.COLOR_BGR2RGB)
                        rgb, int_mat = calibrate(rgb_distorted, [_ for _ in cam_cal.intrinsic], [cam_cal.height, cam_cal.width])
                        # Save cam intrinsics in the first iteration
                        if num_frames == 0:
                            with open(osp.join(traversal_dir, cam_name, img_d_name, 'cam.json'), 'w') as fh: 
                                json.dump({ 'intrinsic' : [_ for _ in cam_cal.intrinsic], 
                                            'dim' : [cam_cal.height, cam_cal.width], 
                                            'extrinsic' : [_ for _ in cam_cal.extrinsic.transform],
                                            'intrinsic_mat': int_mat,
                                            }, fh)
                        # Save images
                        cv2.imwrite(osp.join(traversal_dir, cam_name, img_d_name, 'original', '{:06}.jpg'.format(num_frames)), rgb)
                        down_rgb = cv2.resize(rgb, (rgb.shape[1] // downsample_factor, rgb.shape[0] // downsample_factor), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(osp.join(traversal_dir, cam_name, img_d_name, 'downsample', '{:06}.jpg'.format(num_frames)), down_rgb)

                    if bool_depth:
                        # Get depth map from LiDAR
                        mask = tf.equal(cp_points_all_tensor[..., 0], cam_code)
                        cp_points_all_tensor = tf.cast(tf.gather_nd(cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
                        points_all_tensor = tf.gather_nd(points_all_tensor, tf.where(mask))
                        depth_points = tf.concat([cp_points_all_tensor[..., 1:3], points_all_tensor], axis=-1).numpy()
                        np.save(osp.join(traversal_dir, cam_name, depth_d_name, '{:06}.npy'.format(num_frames)), depth_points)

                    if bool_has_mask:
                        # Get seg mask 
                        cam_mask = cam_masks[cam_name]
                        panoptic_label = camera_segmentation_utils.decode_single_panoptic_label_from_proto(cam_mask)
                        semantic_label, instance_label = camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(panoptic_label, cam_mask.panoptic_label_divisor)

                        # Save semantic and instance label to compressed npz
                        if semantic_label.max() < 256:
                            semantic_label = semantic_label.astype(np.uint8)
                        if instance_label.max() < 256:
                            instance_label = instance_label.astype(np.uint8)
                        np.savez_compressed(osp.join(traversal_dir, cam_name, mask_d_name, '{:06}.npz'.format(num_frames)), semantic=semantic_label, instance=instance_label)
                        
                        # Process individual mask and its corresponding 3D box
                        visible_pcloud = points_all[cp_points_all[:,0]==cam_code]       # list of points that are visible under this camera
                        point_ind_map = np.ones((org_height, org_width, 1)) * -1        # maps ij image coordinate to index of visible_pcloud
                        for ind, (j ,i) in enumerate(cp_points_all[cp_points_all[:,0]==cam_code][:,1:3]):
                            point_ind_map[i,j] = ind
                        point_ind_map = point_ind_map.astype(np.int)                    

                        obj_masks, obj_labels = get_instance_masks(semantic_label, instance_label, labels=list(moveable_categories_mask.keys()))    # individual object masks / labels
                        
                        individual_object_labels = list()
                        for mask, m_lbl in zip(obj_masks, obj_labels):
                            lidar_mask_ind = point_ind_map[(mask>0) & (point_ind_map >= 0)]
                            pcloud_mask = visible_pcloud[lidar_mask_ind]                # list of points that projects within the mask

                            mapped_box = [[None, None, None], [None, None, None], [None, None, None], [None, None, None], [None, None, 0]]
                            for lbl in frame.laser_labels:
                                box = lbl.camera_synced_box 
                                
                                corners = box_utils.get_upright_3d_box_corners(
                                    np.array([[box.center_x, box.center_y, box.center_z, box.length, box.width, box.height, box.heading]])
                                )[0].numpy()                                        # 8 corners - p1,...,p8 : (8,3)

                                frac = get_intersect_fraction(pcloud_mask, corners) # fraction of pcloud_mask in box

                                if frac > mapped_box[4][2]:
                                    meta = lbl.metadata
                                    mapped_box = [  [meta.speed_x, meta.speed_y, meta.speed_z],
                                                    [meta.accel_x, meta.accel_y, meta.accel_z],
                                                    [box.center_x, box.center_y, box.center_z],
                                                    [box.length, box.width, box.height],
                                                    [box.heading, lbl.type, frac],
                                                    ]
                            
                            contours, _ = cv2.findContours((mask[...,0]*255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                            individual_object_labels.append({
                                'mask' : contours,
                                'mask_label' : m_lbl,
                                'speed' : mapped_box[0],
                                'accel' : mapped_box[1],
                                'center' : mapped_box[2],
                                'dim' : mapped_box[3],
                                'heading' : mapped_box[4][0],
                                'box_label' : mapped_box[4][1],
                                'match' : mapped_box[4][2],
                            })
                        
                        with open(osp.join(traversal_dir, cam_name, mask_d_name, '{:06}.pickle'.format(num_frames)), 'wb') as fh:
                            pickle.dump(individual_object_labels, fh)

                    poses[cam_name].append(' '.join([str(x) for x in cam_img.pose.transform]))

                num_frames += 1
                iterator.set_postfix({"Segment": traversal_name, "# Processed": num_frames})

            for cam_name in cam_names:
                out_path = os.path.join(traversal_dir, cam_name, 'odometry.txt')
                with open(out_path, 'w') as fh:
                    for line in poses[cam_name]:
                        fh.write(line + '\n')