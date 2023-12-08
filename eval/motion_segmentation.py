import os.path as osp
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

import sys
proj_dir = osp.dirname(osp.dirname(__file__)) # ./Dynamo-Depth/eval/ -> ./Dynamo-Depth/
sys.path.insert(0, proj_dir)

import torch
from torch.utils.data import DataLoader

from tools import disp_to_depth
from utils import readlines, join_dir, interp, get_model_ckpt_name, is_edge
from Trainer import Trainer
from options import DynamoOptions
import datasets

def main():
    # Hyper-parameters for PR curve calcualtion
    NUM_THRD = 150

    # Override specific options
    options = DynamoOptions()
    opt = options.parse()
    opt.frame_ids = [0, -1, 1]
    opt.print_opt = False                   # suppress command line print out of opt

    # I/O handling
    model_name, ckpt_name = get_model_ckpt_name(opt.load_ckpt)
    outdir = join_dir(opt.eval_dir, f'{model_name}_{opt.dataset}', 'mot_seg')
    pr_curve_path = osp.join(outdir, f'pr_curve_{ckpt_name}.pdf')
    pr_record_path = osp.join(outdir, f'pr_record_{ckpt_name}.npz')
    fp_tally_path = osp.join(outdir, f'fp_tally_{ckpt_name}.pdf')

    # Initialize the trainer object
    trainer = Trainer(opt)
    trainer.set_eval()

    # Initialize the dataloader
    filenames = readlines(osp.join(proj_dir, 'splits', opt.split, 'test_mask_files.txt'))
    # prune edge cases: evaluated frames must not be the first or last frame in the video sequence
    filenames = [f for f in filenames if not is_edge(f, opt)]
    assert len(filenames) > 0, 'Number of items for eval must be > 0.'
    dataset = trainer.get_dataset(filenames, is_train=False, load_depth=False, load_mask=True)
    dataset.img_type = opt.eval_img_type
    full_width, full_height = dataset.full_res_shape
    loader = DataLoader(dataset, opt.batch_size, False, num_workers=opt.num_workers, pin_memory=True, drop_last=False)
    print(f'=== len={len(dataset)} ===')
    
    # Setup data collections
    eps = 1/(NUM_THRD-1)
    thrds = torch.linspace(0-eps, 1-eps, NUM_THRD).reshape(1, NUM_THRD, 1, 1).to(trainer.device)
    motion_pred = [None for i in range(len(dataset))]
    record = {val : torch.zeros(NUM_THRD).to(trainer.device) for val in ['tp', 'fp', 'fn']}

    # Evaluate
    with torch.no_grad():
        for batch_idx, inputs in tqdm(enumerate(loader), desc='(1/2) Calculating motion / scanning thresholds', total=len(loader)):
            
            # Get predictions via trainer
            trainer.process_inputs(inputs)
            outputs = trainer.model(inputs)

            # Get binary predicted motion mask at different thresholds
            pred_mask = interp(outputs[('motion_mask',-1,0)], (full_height, full_width))            # (B, 1, H, W)               
            pred_mask_b = pred_mask > thrds                                                         # (B, NUM_THRD, H, W)               
            
            # Get ground truth motion segmentation | 1=moving object, 2=static object, 3=unlabeled
            gt_mask = inputs['mot_mask'].unsqueeze(1)                                               # (B, 1, H, W)   
            gt_mask_b = gt_mask == 1
            valid_b = (gt_mask != 3).int()

            # Store predicted masks to be used later for false positive tally
            for ii, ind in enumerate(inputs['index'].tolist()):
                motion_pred[ind] = pred_mask[ii][0].cpu().numpy()

            # Increment true positive, false positive, and false negative
            for bi in range(gt_mask_b.size(0)):

                vm = valid_b[bi]                    # valid mask :      (1, H, W)
                gm = gt_mask_b[bi]                  # gt mask:          (1, H, W)
                pm = pred_mask_b[bi]                # pred mask:        (NUM_THRD, H, W)
                inter = torch.logical_and(gm, pm)   # intersect:        (NUM_THRD, H, W)

                g_sum = gm.sum((1,2))               # num gt:           (1,)
                p_sum = (pm*vm).sum((1,2))          # num pred:         (NUM_THRD,)   # disregard prediction if the region is not valid

                tp = inter.sum((1,2))               # true positive:    (NUM_THRD,)
                fp = p_sum - tp                     # false positive:   (NUM_THRD,)
                fn = g_sum - tp                     # false negative:   (NUM_THRD,)

                record['tp'] += tp
                record['fp'] += fp
                record['fn'] += fn

    # Compute precision, recall, and f1 per each threshold
    precision = (record['tp'] / (record['tp'] + record['fp'] + 1e-10)).cpu().numpy()
    recall = (record['tp'] / (record['tp'] + record['fn'] + 1e-10)).cpu().numpy()
    f1 = 2 * (precision*recall) / (precision+recall + 1e-10)

    # Plot Precision-Recall Curve and save data
    fig = plt.figure()
    plt.axhline(y = precision[0], linestyle = ':', color=f'C{ii}') # baseline
    plt.plot(recall[recall>0], precision[recall>0], color=f'C{ii}')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Motion Segmentation PR Curve')
    fig.savefig(pr_curve_path)
    plt.clf()
    out_npz = {'precision' : precision, 'recall' : recall, 'f1' : f1, 'thrds' : thrds.cpu().numpy()}
    np.savez(pr_record_path, **out_npz)
    print(f'PR curve saved to `{pr_curve_path}`.')
    print(f'PR record saved to `{pr_record_path}`.')

    if opt.dataset == 'waymo':  # waymo has semantic labels, nuscenes does not
        # Iterate over predictions and find false positive ranking with best threshold
        best_thrd_idx = np.argmax(f1)
        best_f1_thrd = thrds.squeeze()[best_thrd_idx].item()
        fp_tally = {'total' : 0}
        for batch_idx, inputs in tqdm(enumerate(loader), desc='(2/2) Scanning false positives                ', total=len(loader)):
            mot_mask_gt = inputs['mot_mask'].numpy()    # used to decide true/false positives
            sem_mask_gt = inputs['sem_mask'].numpy()    # used to assign semantic classes

            for ii, ind in enumerate(inputs['index'].tolist()):

                gt_mask_b = mot_mask_gt[ii] == 1        # binary moving mask:           (H, W)
                valid_b = mot_mask_gt[ii] != 3          # binary valid (labeled) mask:  (H, W)
                sem_mask = sem_mask_gt[ii]              # semantic mask:                (H, W)
                pred_mask = motion_pred[ind]            # predicted mask (range 0->1):  (H, W)

                # find pixels that are false positives and their associated semantic class
                pred_mask_b = pred_mask > best_f1_thrd           
                fp_b = np.logical_and(pred_mask_b > gt_mask_b, valid_b)

                for label, count in zip(*np.unique(sem_mask[fp_b], return_counts=True)):
                    fp_tally[label] = fp_tally[label] + count if label in fp_tally else count
                    fp_tally['total'] += count
        
        # Plot False Positive Tally 
        fig = plt.figure()
        fig.set_size_inches(20, 10)
        cats, cnts = [], []
        for c_idx, cnt in fp_tally.items():
            if c_idx != 'total':
                cats.append(dataset.categories[c_idx])
                cnts.append(cnt / fp_tally['total'])
        sort_ind = np.argsort(cnts)[::-1]
        plt.bar(np.array(cats)[sort_ind], np.array(cnts)[sort_ind])
        plt.tick_params(axis='x', labelrotation = 60)
        plt.ylim([0,1])
        plt.ylabel('False Positive Rate')
        plt.title('Motion Segmentation False Positive Tally - Thrd {:.2f} - Macro F1 {:.3f}'.format(best_f1_thrd, np.max(out_npz['f1'])))
        fig.savefig(fp_tally_path)
        print(f'FP tally saved to `{fp_tally_path}`.')

if __name__ == '__main__':
    main()

