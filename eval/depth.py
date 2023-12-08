import os.path as osp
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import sys
proj_dir = osp.dirname(osp.dirname(__file__)) # ./Dynamo-Depth/eval/ -> ./Dynamo-Depth/
sys.path.insert(0, proj_dir)

from Trainer import Trainer
from options import DynamoOptions
from tools import disp_to_depth
from utils import readlines, write_to_file, join_dir, get_model_ckpt_name
import datasets

def display_str(l):
    return ''.join(['{:^15s}'.format(m) for m in l])

def main():
    # Override specific options
    options = DynamoOptions()
    opt = options.parse()
    opt.print_opt = False           # suppress command line print out of opt
    opt.frame_ids = [0]             # only need to process one frame
    opt.img_ext = opt.eval_img_ext

    # I/O handling
    model_name, ckpt_name = get_model_ckpt_name(opt.load_ckpt)
    outdir = join_dir(opt.eval_dir, f'{model_name}_{opt.dataset}', 'depth')
    out_path = osp.join(outdir, f'{ckpt_name}.txt')
    out_strings = list()    # used to write to terminal + file
    
    # Initialize the trainer and depth_metric object (no need to run the independent motion networks)
    trainer = Trainer(opt)
    trainer.base_model.bool_CmpFlow = False
    trainer.base_model.bool_MotMask = False
    trainer.set_eval()
    depth_metrics = trainer.depth_metrics           # used to compute depth metrics

    metric_names = depth_metrics.depth_metric_names
    header_str = display_str(['Split'] + metric_names)
    out_strings.append(f'====== Model Path - {opt.load_ckpt} ======\n')

    ### Part 1 - Compute overall depth performance
    out_strings.append('====== Depth Eval on Overall Test Set ======\n')

    # Initialize the dataloader
    filenames = readlines(osp.join(proj_dir, 'splits', opt.split, 'test_files.txt'))
    assert len(filenames) > 0, 'Number of items for eval must be > 0.'
    dataset = trainer.get_dataset(filenames, is_train=False, load_depth=True, load_mask=False)
    dataset.img_type = opt.eval_img_type
    loader = DataLoader(dataset, opt.batch_size, False, num_workers=opt.num_workers, pin_memory=True, drop_last=False)
    out_strings.append(f'=== len={len(dataset)} ===')
    out_strings.append(header_str)

    # Iterate over examples
    metrics = {m:0 for m in metric_names}
    total_num = 0
    with torch.no_grad():
        for batch_idx, inputs in tqdm(enumerate(loader), desc='(1/2) Computing Overall Depth Metrics       ', total=len(loader)):
            
            # Get predictions via trainer
            trainer.process_inputs(inputs)
            outputs = trainer.model(inputs)
            batch_cnt = outputs[('disp', 0, 0)].size(0) # since batch_size can be >1 and drop_last=False
            
            # Compute and aggregate depth performance metrics
            outputs[('disp_scaled', 0, 0)], _ = disp_to_depth(outputs[('disp', 0, 0)], opt.min_depth, opt.max_depth)
            met = depth_metrics(inputs, outputs)
            for m in metric_names:
                metrics[m] += met[m].item() * batch_cnt
            total_num += batch_cnt
    
    out_strings.append(display_str(['OVERALL'] + ['& {:.3f}'.format(metrics[m] / total_num) for m in metrics]))
    out_strings.append('\n')


    ### Part 2 - Compute mask-dependent depth performance

    out_strings.append('====== Depth Eval on Test Set with Segmentation Annotations ======\n')
    mask_split_path = osp.join(proj_dir, 'splits', opt.split, 'test_mask_files.txt')
    if opt.dataset == 'kitti':
        out_strings.append('Mask Split Evaluation Skipped for KITTI.')
    else:
        # Initialize the dataloader
        filenames = readlines(mask_split_path)
        assert len(filenames) > 0, 'Number of items for eval must be > 0.'
        dataset = trainer.get_dataset(filenames, is_train=False, load_depth=True, load_mask=True)
        dataset.img_type = opt.eval_img_type
        loader = DataLoader(dataset, opt.batch_size, False, num_workers=opt.num_workers, pin_memory=True, drop_last=False)
        out_strings.append(f'=== len={len(dataset)} ===')
        out_strings.append(header_str)

        # Iterate over examples
        labels = {'bg' : 0, 'static' : 2, 'mot' : 1}    # semantic splits
        metrics = {split:{m:[0,0] for m in metric_names} for split in labels.keys()}
        with torch.no_grad():
            for batch_idx, inputs in tqdm(enumerate(loader), desc='(2/2) Computing Mask-Dependent Depth Metrics', total=len(loader)):
                
                # Get predictions via trainer
                trainer.process_inputs(inputs)
                outputs = trainer.model(inputs)

                # Compute and aggregate depth performance metrics
                outputs[('disp_scaled', 0, 0)], _ = disp_to_depth(outputs[('disp', 0, 0)], opt.min_depth, opt.max_depth)
                met = depth_metrics(inputs, outputs, mask=inputs['mot_mask'])  # condition on the motion masks
                for split in labels.keys(): 
                    for m in metric_names:
                        if labels[split] in met[f'{m}_mask']:
                            # tally to compute macro average
                            metrics[split][m][0] += met[f'{m}_mask'][labels[split]][0]  # metric * cnt
                            metrics[split][m][1] += met[f'{m}_mask'][labels[split]][1]  # cnt

        for split in labels.keys():
            out_strings.append(display_str( [split.upper()] + ['& {:.3f}'.format(metrics[split][m][0] / metrics[split][m][1]) for m in metrics[split]] ))
        out_strings.append('\n')

    # Write to terminal / out_path
    for s in out_strings:
        print(s)
    write_to_file(out_strings, out_path)

if __name__ == '__main__':
    main()
