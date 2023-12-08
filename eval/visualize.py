import os.path as osp
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

import os, sys
proj_dir = osp.dirname(osp.dirname(__file__)) # ./Dynamo-Depth/eval/ -> ./Dynamo-Depth/
sys.path.insert(0, proj_dir)

from tools import disp_to_depth
from utils import make_mp4, readlines, hsv_to_rgb, join_dir, score_map_vis, get_model_ckpt_name, get_filenames, is_edge
from networks.layers import transformation_from_parameters

from Trainer import Trainer
from options import DynamoOptions

def get_rgb_np(img):
    """ Convert given img tensor (1,3,H,W) into numpy
    """
    return img[0].permute(1,2,0).cpu().numpy()

def get_vis(opt, trainer, inputs, ref_frame_id, scale=0, items=('img', 'disp', 'ego_flow', 'ind_flow', 'mask')):
    """ Process a given batch and produce raw visualizations given by items
    """
    s = scale
    f_id = ref_frame_id

    with torch.no_grad():
        trainer.apply_img_resize(inputs)
        for key, inp in inputs.items():
            inputs[key] = inp.to(trainer.device) if key != 'paths' else inp
        outputs = trainer.model(inputs)
    
    collection = dict()

    # Target Image
    if 'img' in items: 
        collection['img'] =     inputs[('color', 0, s)]                 # (B, 3, H, W)
    
    # Source Image
    if 'ref_img' in items: 
        collection['ref_img'] = inputs[('color', f_id, s)]              # (B, 3, H, W)
    
    # Disparity
    if 'disp' in items: 
        collection['disp'] =    outputs[('disp', 0, s)]                 # (B, 1, H, W)
    
    # Binary Motion Mask
    if 'mask' in items: 
        collection['mask'] =    outputs[('motion_mask', f_id, s)]          # (B, 1, H, W)

    if any(['flow' in it for it in items]):
        _, depth = disp_to_depth(outputs[('disp', 0, s)], opt.min_depth, opt.max_depth)
        K, inv_K =  inputs[('K', s)], inputs[('inv_K', s)]              # (B, 4, 4), (B, 4, 4)
        axisangle = outputs[('axisangle', 0, f_id)]
        translation = outputs[('translation', 0, f_id)]
        time_step = inputs[('ts', f_id)].reshape(-1,1,1).float()        # (B, 1, 1) time step from 0 to f_id - from dataloader, never close to 0
        camTcam = transformation_from_parameters(axisangle/time_step, translation/time_step, invert=True) # (B, 4, 4)
            
        # Ego-motion / Rigid Flow
        if 'ego_flow' in items: 
            _, ego_flow_hsv, ego_flow_mag = trainer.vis_motion(depth=depth, K=K, inv_K=inv_K, motion_map=None, camTcam=camTcam, scale=s)                # (1, 3, H, W), float
            collection['ego_flow'] = {'hsv' : ego_flow_hsv, 'mag' : ego_flow_mag}

        # Independent Flow
        if 'ind_flow' in items or 'samp_flow' in items:
            cam_points = trainer.backproject_depth[s](depth, inv_K)
            _, ego_flow = trainer.project_3d[s](cam_points, K, camTcam) 
            independ_flow = outputs[('motion_mask', f_id, s)] * (outputs[('complete_flow', f_id, s)] - ego_flow.reshape(-1, 3, opt.height, opt.width))   # (B, 3, H, W)
            _, ind_flow_hsv, ind_flow_mag = trainer.vis_motion(depth=depth, K=K, inv_K=inv_K, motion_map=independ_flow, camTcam=None, scale=s)          # (1, 3, H, W), float        
            collection['ind_flow'] = {'hsv' : ind_flow_hsv, 'mag' : ind_flow_mag}
        
        # Complete Flow
        if 'comp_flow' in items:
            complete_flow = outputs[('complete_flow', f_id, s)]
            _, comp_flow_hsv, comp_flow_mag = trainer.vis_motion(depth=depth, K=K, inv_K=inv_K, motion_map=complete_flow, camTcam=None, scale=s)        # (1, 3, H, W), float
            collection['comp_flow'] = {'hsv' : comp_flow_hsv, 'mag' : comp_flow_mag}
        
        # Ego-motion / Rigid Flow + Independent Flow
        if 'samp_flow' in items:
            _, samp_flow_hsv, samp_flow_mag = trainer.vis_motion(depth=depth, K=K, inv_K=inv_K, motion_map=independ_flow, camTcam=camTcam, scale=s)     # (1, 3, H, W), float      
            collection['samp_flow'] = {'hsv' : samp_flow_hsv, 'mag' : samp_flow_mag}
        
    return collection

def combine_vis(vis_list, arrangement, consistent_flow=True, flow_mag_factor=1.0, mask_max_mag=1.0):
    """ aggregate visualizations into an image according to the arrangement
    """
    vis_frames = list()

    if consistent_flow and any(['flow' in a for arr in arrangement for a in arr]):
        max_flow_mag = max( [max([vis[a]['mag'] for arr in arrangement for a in arr if 'flow' in a]) for vis in vis_list] )

    for vis in vis_list:
        to_vstack = list()
        for arr in arrangement: # 2d arrangement
            to_hstack = list()
            for a in arr:
                out = vis[a]
                if 'img' in a:
                    out = get_rgb_np(out)
                elif a == 'mask':
                    out = score_map_vis(out, 'hot', vminmax=(0,mask_max_mag))
                elif a == 'disp':
                    out = score_map_vis(out, 'plasma', vminmax=(0,1))
                elif 'flow' in a:
                    # makes small motion vectors more visible if max_flow_mag < 1
                    if consistent_flow: 
                        max_mag = flow_mag_factor * max_flow_mag
                    else:
                        max_mag = flow_mag_factor * max([vis[a]['mag'] for arr in arrangement for a in arr if 'flow' in a])
                    
                    hsv = vis[a]['hsv']
                    hsv[:, 2] = torch.clamp(hsv[:, 2] * vis[a]['mag'] / max_mag, 0, 1)
                    out = get_rgb_np(1 - hsv_to_rgb(hsv))
                else:
                    raise Exception(f'Arrangement name (={a}) not recognized.')
                
                to_hstack.append((out*255).astype(np.uint8))
            to_vstack.append(np.hstack(to_hstack))
        vis_frames.append(np.vstack(to_vstack))
    return vis_frames

def vis_segment(opt, trainer, val_segment, outdir):
    """ Predict for a single image or folder of images
    """
    # used to define what is visualized + how they are arranged
    arrangement = [
        ['img', 'disp', 'ego_flow', 'ind_flow', 'mask'],
    ]

    # Initialize the dataloader
    filenames = [f for f in get_filenames(val_segment, opt) if not is_edge(f, opt)]
    dataset = trainer.get_dataset(filenames, is_train=False, load_depth=False, load_mask=False, path=True)
    dataset.img_type = opt.eval_img_type
    loader = DataLoader(dataset, 1, False, num_workers=opt.num_workers, pin_memory=True, drop_last=False)

    # Iterate over the sequence
    vis_list = [dict() for i in range(len(loader))]
    for batch_idx, inputs in tqdm(enumerate(loader), desc='Computing per-frame predictions', total=len(loader)):

        frame_vis = get_vis(opt, trainer, inputs, ref_frame_id=opt.frame_ids[1], scale=0, items=arrangement[0])
        f_index = int(inputs['paths'][1][0]) - 1
        vis_list[f_index].update(frame_vis)
    
    out_frames = combine_vis(vis_list, arrangement)
    
    out_vid_name = osp.join(outdir, '{}.mp4'.format(val_segment.split('/')[1]))
    fps = 13 if opt.dataset == "nuscenes" else 10   # dataset info 
    make_mp4(out_frames, out_vid_name, fps=fps, bgr=False)
    print(f'Saved to `{out_vid_name}`\n')


def main():
    # Override specific options
    options = DynamoOptions()
    opt = options.parse()
    opt.num_workers = 1
    opt.batch_size = 1
    opt.print_opt = False       # suppress command line print out of opt    

    # I/O handling
    model_name, ckpt_name = get_model_ckpt_name(opt.load_ckpt)
    outdir = join_dir(opt.eval_dir, f'{model_name}_{opt.dataset}', 'vis', ckpt_name)

    # Initialize the trainer object
    trainer = Trainer(opt)
    trainer.set_eval()
    trainer.setup_phase('fine_tune')    # assuming all modules are trained / no model is just initialized that needs to be turned off

    # Get segments to visualize
    files = readlines(osp.join(proj_dir, 'splits', opt.split, 'test_files.txt'))
    segments = sorted(list(set([f.split()[0] for f in files])))

    # Iterate over the segments
    for ii, segment in enumerate(segments):
        print(f"{ii+1}/{len(segments)} segments - {segment}")
        vis_segment(opt, trainer, segment, outdir)

if __name__ == '__main__':
    main()

