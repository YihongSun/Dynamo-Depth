import os
from tqdm import tqdm
import os.path as osp
import numpy as np

import sys
proj_dir = osp.dirname(osp.dirname(__file__)) # ./Dynamo-Depth/eval/ -> ./Dynamo-Depth/
sys.path.insert(0, proj_dir)

import torch
from torch.utils.data import DataLoader

from networks.layers import transformation_from_parameters
from utils import readlines, join_dir, write_to_file, get_model_ckpt_name, get_filenames, is_edge
from Trainer import Trainer
from options import DynamoOptions

# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs


# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):
    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse


def eval_odom(opt, trainer, val_segment, track_length):
    """Function to predict for a single image or folder of images
    """

    # Initialize the dataloader
    filenames = [f for f in get_filenames(val_segment, opt) if not is_edge(f, opt)]
    dataset = trainer.get_dataset(filenames, is_train=False, load_depth=False, load_mask=False)
    dataset.img_type = opt.eval_img_type
    loader = DataLoader(dataset, 1, False,num_workers=opt.num_workers, pin_memory=True, drop_last=False)
    N = len(filenames)

    # Iterate
    f_id = -1; s = 0
    pred_poses = []
    for batch_idx, inputs in enumerate(loader):
        with torch.no_grad():
            trainer.apply_img_resize(inputs)
            for key, inp in inputs.items():
                inputs[key] = inp.to(trainer.device)

            outputs = trainer.model(inputs)
            translation = outputs[('translation', 0, 1)]
            axisangle = outputs[('axisangle', 0, 1)]
            pred_poses.append(transformation_from_parameters(axisangle, translation).cpu().numpy())
    pred_poses = np.concatenate(pred_poses)

    # Get ground-truth poses
    gt_poses_path = osp.join(opt.data_path, val_segment, opt.cam_name, 'odometry.txt')
    gt_global_poses = np.loadtxt(gt_poses_path)[1:] # ignore the first frame
    assert N == gt_global_poses.shape[0]-1
    gt_global_poses = gt_global_poses.reshape(N+1, -1, 4)
    if gt_global_poses.shape[1] == 3:
        gt_global_poses = np.concatenate((gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)
        gt_global_poses[:, 3, 3] = 1
    gt_xyzs = gt_global_poses[:, :3, 3]
    gt_local_poses = []
    for i in range(1, len(gt_global_poses)):
        gt_local_poses.append(np.linalg.inv(np.dot(np.linalg.inv(gt_global_poses[i - 1]), gt_global_poses[i])))

    # Evaluate ates and speeds
    ates = []
    speeds = []
    num_frames = gt_xyzs.shape[0]
    for i in range(0, num_frames - 1):
        local_xyzs = np.array(dump_xyz(pred_poses[i:i + track_length - 1]))
        gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i:i + track_length - 1]))
        if local_xyzs.shape[0] < track_length - 1:
            continue
        local_xyzs = np.concatenate((local_xyzs[:,2:3],local_xyzs[:,0:1], local_xyzs[:,1:2]), 1)    # shift axis around
        ates.append(compute_ate(gt_local_xyzs, local_xyzs))
        speeds.append(np.sqrt(((gt_local_xyzs[1:] - gt_local_xyzs[:-1]) ** 2).sum(1)).mean())

    return ates, speeds

def main():
    # Hyper-parameters for evaluation
    track_length = 5
    stop_segment = 100

    # Override specific options
    options = DynamoOptions()
    opt = options.parse()
    opt.frame_ids = [0, -1, 1]
    opt.print_opt = False                   # suppress command line print out of opt
    opt.num_workers = 1
    opt.batch_size = 1
    assert opt.dataset == 'waymo' or opt.dataset == 'nuscenes', f'Only implemented for waymo and nuscenes, {opt.dataset} is not supported.'

    # I/O handling
    model_name, ckpt_name = get_model_ckpt_name(opt.load_ckpt)
    outdir = join_dir(opt.eval_dir, f'{model_name}_{opt.dataset}', 'odometry')
    txt_path = osp.join(outdir, f'record_{ckpt_name}-{track_length}.txt')
    npy_path = osp.join(outdir, f'record_{ckpt_name}-{track_length}.npy')
    
    # Initialize the trainer object
    trainer = Trainer(opt)
    trainer.set_eval()

    # Get segments to visualize
    files = readlines(osp.join(proj_dir, 'splits', opt.split, 'test_files.txt'))
    val_segments = sorted(list(set([f.split()[0] for f in files])))[:stop_segment]

    # Iterate
    output_strs = [f'=== track_length: {track_length}']
    all_ates = []
    all_speeds = []
    for ii, val_segment in tqdm(enumerate(val_segments), desc='Evaluating segments', total=len(val_segments)):
        ates, speeds = eval_odom(opt, trainer, val_segment, track_length)
        all_ates += ates
        all_speeds += speeds

        out_str = f'{val_segment:50s} Track={track_length} ATE: {np.mean(ates):0.3f} ± {np.std(ates):0.3f},  Speed: {np.mean(speeds):0.3f} ± {np.std(speeds):0.3f},  Len: {len(all_ates)}'
        output_strs.append(out_str)
    
    # Results
    output_strs.append(f'\nATE Trajectory error (Track={track_length}):  ')
    output_strs.append(f'Mean:   {np.mean(all_ates)}')
    output_strs.append(f'std:    {np.std(all_ates)}')
    output_strs.append('--')
    output_strs.append(f'Min:    {np.min(all_ates)}')
    output_strs.append(f'Median: {np.median(all_ates)}')
    output_strs.append(f'Max:    {np.max(all_ates)}')
    
    output_strs.append('==')
    output_strs.append('\nSpeed:  ')
    output_strs.append(f'Mean:   {np.mean(all_speeds)}')
    output_strs.append(f'std:    {np.std(all_speeds)}')
    output_strs.append('--')
    output_strs.append(f'Min:    {np.min(all_speeds)}')
    output_strs.append(f'Median: {np.median(all_speeds)}')
    output_strs.append(f'Max:    {np.max(all_speeds)}')
    output_strs.append('--')
    output_strs.append(f'len:    {len(all_speeds)}')

    # Write to terminal / out_path
    for s in output_strs:
        print(s)
    write_to_file(output_strs, txt_path)
    np.save(npy_path, np.stack((np.array(all_ates), np.array(all_speeds))).transpose((1,0)))

if __name__ == '__main__':
    main()
