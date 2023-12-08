import os, json, cv2, time
import os.path as osp
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from utils import readlines, join_dir, interp, make_ind_map, cart2polar, hsv_to_rgb, sec_to_hm_str
from tools import DepthMetrics, GroundPlane, BackprojectDepth, Project3D, SSIM, disp_to_depth, depth_to_disp, compute_smooth_loss
import datasets
import networks

class Trainer:
    def __init__(self, options):
        
        ## Assertions

        self.opt = options
        assert self.opt.height % 32 == 0, f'height(={self.opt.height}) must be a multiple of 32'
        assert self.opt.width % 32 == 0, f'width(={self.opt.width}) must be a multiple of 32'
        assert self.opt.frame_ids[0] == 0, f'frame_ids(={self.opt.frame_ids}) must start with 0'
        assert len(self.opt.epoch_schedules) == 4 and all([e >= 0 for e in self.opt.epoch_schedules]), f'epoch_schedules(={self.opt.epoch_schedules}) must be length=4 and non-negative'

        self.local_rank = self.opt.local_rank
        self.cuda_id = self.opt.cuda_ids[self.local_rank]
        assert self.cuda_id < torch.cuda.device_count(), f'cuda_ids[local_rank](={self.cuda_id}) must be visible'
        self.device = torch.device("cuda:{}".format(self.cuda_id) if torch.cuda.is_available() else "cpu")

        self.print('\n=============== Trainer Initialization ===============')

        ## Setup model

        self.base_model = networks.Model(self.opt)
        if self.opt.load_ckpt != "":
            self.load_model()
        self.base_model.to(self.device)
        
        self.model = DDP(self.base_model, device_ids=[self.cuda_id], find_unused_parameters=True) if self.opt.ddp else self.base_model
        self.model.to(self.device)
        
        ## Setup Trainer

        self.num_scales = len(self.opt.scales)
        self.B = self.opt.batch_size
        self.H = self.opt.height
        self.W = self.opt.width   

        self.log_path = osp.join(self.opt.log_dir, self.opt.model_name)
        
        ## Setup Dataset

        datasets_dict = {'kitti': datasets.KITTIDataset, 'waymo': datasets.WaymoDataset, 'nuscenes': datasets.nuScenesDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        # Setup Metrics / Auxillary Tools

        self.depth_metrics = DepthMetrics(self.opt.eval_img_bound, self.opt.eval_min_depth, self.opt.eval_max_depth)
        self.gplane = GroundPlane(num_points_per_it=self.opt.gp_np_per_it, max_it=self.opt.gp_max_it, tol=self.opt.gp_tol, g_prior=self.opt.gp_prior)
        self.ssim = SSIM().to(self.device)
        self.bce = nn.BCEWithLogitsLoss()
        self.I = torch.eye(4).reshape(1,4,4).repeat(self.B,1,1).to(self.device)
        
        # Setup Scale-Dependent Tools

        self.resize = {}
        self.backproject_depth = {}
        self.project_3d = {}
        self.prob_target = {}

        for scale in self.opt.scales:
            h = self.H // (2 ** scale)
            w = self.W // (2 ** scale)

            self.resize[scale] = torchvision.transforms.Resize((h,w),interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True)
            self.backproject_depth[scale] = BackprojectDepth(self.B, h, w).to(self.device)
            self.project_3d[scale] = Project3D(self.B, h, w).to(self.device)
            self.prob_target[scale] = torch.zeros(self.B, 1, h, w).to(self.device)
        
        self.save_opt()
        self.print('=============== Trainer Initialization ===============\n')

    # ========== Standard functions ========== #

    def train(self):
        """ Execute the entire training pipeline according to the self.opt.epoch_schedules
        """
        
        self.setup_wandb()
        self.g_step = 0             # global step count
        self.init_loaders()
        
        for phase_i, phase_name in enumerate(['disp_init', 'motion_init', 'mask_init', 'fine_tune']):

            num_epoch = self.opt.epoch_schedules[phase_i]
            self.print(f'======== {phase_name.upper()} - Num Epochs={num_epoch} ========')

            if num_epoch > 0:
                self.run_phase(phase_name, num_epoch)

            self.print(f'======== {phase_name.upper()} - Num Epochs={num_epoch} ========\n')

    def run_phase(self, phase_name, num_epoch):
        """ Execute a particular phase according to the phase_name and the associated num_epoch
        """

        self.setup_phase(phase_name)
        
        self.step = 0
        self.epoch = 0
        
        self.bool_automask = phase_name == 'disp_init' # turn on automasking only during `disp_init`
        self.num_total_steps = self.num_steps_per_epoch * num_epoch

        self.start_time = time.time()
        for self.epoch in range(num_epoch):
            self.print()
            self.run_epoch()
            if ((self.epoch + 1) % self.opt.save_frequency == 0) or (self.epoch == num_epoch - 1):
                self.save_model(phase_name)

    def run_epoch(self):
        """ Run a single epoch of training and validation 
        """
        
        # reset train loader
        self.setup_train_loader()    
        self.set_train()

        gpu_time, data_loading_time = 0, 0
        before_op_time = time.time()

        self.optim['optimizer'].zero_grad()
        
        for batch_idx, inputs in enumerate(self.train_loader):
            
            data_loading_time += (time.time() - before_op_time)
            before_op_time = time.time()

            # === Compute Starts === #
            
            outputs, losses = self.process_batch(inputs)
            losses['loss'].backward()
            
            self.optim['optimizer'].step()
            self.optim['optimizer'].zero_grad()
                
            # === Compute Ends === #

            compute_duration = time.time() - before_op_time
            gpu_time += compute_duration

            early_freq = self.opt.log_frequency
            late_freq = 10 * early_freq
            early_phase = batch_idx % early_freq == 0 and self.step < late_freq
            late_phase = self.step % late_freq == 0
            if early_phase or late_phase:
                self.log_time(batch_idx, compute_duration, losses['loss'].cpu().data, data_loading_time, gpu_time)
                gpu_time, data_loading_time = 0, 0
                self.log('train', inputs, outputs, losses)
                self.val(batch_idx)

            del outputs
            self.g_step += 1
            self.step += 1
            before_op_time = time.time()

        self.optim['lr_scheduler'].step()

    def val(self, batch_idx):
        """ Validate the model on a single minibatch
        """

        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if self.val_loader.dataset.load_depth:
                losses.update(self.depth_metrics(inputs, outputs))

            self.log('val', inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def process_batch(self, inputs):
        """ Pass a minibatch through the network and generate images and losses 
        """
        
        # === Process inputs - resize and device loading === #
        self.process_inputs(inputs)
        
        # === Calculate depth + egomotion + independent motion === #
        outputs = self.model(inputs)

        # === Warp and calculate loss ==== #
        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    # ========== Loss functions ========== #

    def generate_images_pred(self, inputs, outputs):
        """ Generate the warped (reprojected) color images for a minibatch, which are saved in outputs. 
        """

        for scale in self.opt.scales:

            source_scale = 0
            h, w = outputs[('disp', 0, scale)].shape[-2:]
            B = outputs[('disp', 0, scale)].size(0)

            disp = interp(outputs[('disp', 0, scale)], (self.H, self.W))
            
            disp_scaled, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[('depth', 0, scale)] = depth
            outputs[('disp_scaled', 0, scale)] = disp_scaled

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                
                K = inputs[('K', source_scale)]
                T = outputs[('cam_T_cam', 0, frame_id)]

                cam_points = self.backproject_depth[source_scale](depth, inputs[('inv_K', source_scale)])
                outputs[('cam_points', 0, scale)] = cam_points

                # === Compute motion mask === #
                if self.base_model.bool_MotMask:
                    outputs[('motion_mask_r', frame_id, scale)] = interp(outputs[('motion_mask', frame_id, scale)], (self.H, self.W))   # resize to original shape
                else:
                    outputs[('motion_mask', frame_id, scale)] = torch.ones(B, 1, h, w).to(self.device)
                    outputs[('motion_mask_r', frame_id, scale)] = torch.ones(B, 1, self.H, self.W).to(self.device)

                # === Compute sample for reconstruction === #
                if self.base_model.bool_CmpFlow: # complete 3D flow is predicted
                    # compute 3D flows
                    sample_ego, ego_flow = self.project_3d[source_scale](cam_points, K, T) # (B, H, W, 2), (B, 3, H*W)
                    complete_flow = interp(outputs[('complete_flow', frame_id, scale)], (self.H, self.W)).view(B,3,-1) * inputs[('ts', frame_id)].view(B, 1, 1)
                    residual_flow = complete_flow - ego_flow
                    independ_flow = residual_flow * outputs[('motion_mask_r', frame_id, scale)].view(B, 1, -1)

                    # compute 2D samples - detached since they are only used for motion mask supervision
                    outputs[('sample_ego', frame_id, scale)] = sample_ego.detach()
                    cam_points_tmp = cam_points.detach().clone()
                    cam_points_tmp[:,:3] += complete_flow
                    sample_complete, _ = self.project_3d[source_scale](cam_points_tmp, K, T=None)
                    outputs[('sample_complete', frame_id, scale)] = sample_complete.detach()

                    if self.base_model.bool_MotMask:
                        # Project into 3D via inv_K again - second pass for flow calculation
                        # only add the contribution of independent flow, transformation is applied afterwards
                        cam_points = self.backproject_depth[source_scale](depth, inputs[('inv_K', source_scale)])
                        cam_points[:,:3] += independ_flow
                        sample, _ = self.project_3d[source_scale](cam_points, K, T)
                    else:
                        # since only learning the complete flow, it is added without using any transformation
                        cam_points[:,:3] += complete_flow
                        sample, _ = self.project_3d[source_scale](cam_points, K, T=None) # (B, H, W, 2), (B, 1, H, W), (B, 3, H*W)

                else:                           # complete 3D flow is not predicted     
                    # compute 3D flows
                    sample, ego_flow = self.project_3d[source_scale](cam_points, K, T) # (B, H, W, 2), (B, 1, H, W), (B, 3, H*W)
                    residual_flow = torch.zeros_like(ego_flow)
                    independ_flow = torch.zeros_like(ego_flow)
                    # Since complete 3D flow is not predicted, 2D samples sample_ego and sample_complete are the same, hence not recorded

                outputs[('sample', frame_id, scale)] = sample
                outputs[('color', frame_id, scale)] = F.grid_sample(inputs[('color', frame_id, source_scale)], sample, padding_mode='border', align_corners=True)
                outputs[('ego_flow', frame_id, scale)] = ego_flow
                outputs[('independ_flow', frame_id, scale)] = independ_flow.reshape(B, 3, self.H, self.W)
                outputs[('residual_flow', frame_id, scale)] = interp(residual_flow.reshape(B, 3, self.H, self.W), (h, w))

                if self.bool_automask:
                    outputs[('color_identity', frame_id, scale)] = inputs[('color', frame_id, source_scale)]

    def compute_losses(self, inputs, outputs):
        """ Compute the reprojection and smoothness losses for a minibatch
        """

        move_Depth   = 'Depth' in self.optim['network_names']
        move_CmpFlow = 'CmpFlow' in self.optim['network_names']
        move_MotMask = 'MotMask' in self.optim['network_names']

        source_scale = 0
        losses = {'loss' : 0}
        loss_terms = [k[2:] for k in self.opt.__dict__.keys() if 'g_' == k[:2]]
        for term in loss_terms + self.opt.scales:
            losses[f'loss_term/{term}'] = 0
        
        for loss_term in loss_terms:
            coef_name = 'g_' + loss_term
            loss_val = getattr(self.opt, coef_name)

            if coef_name in self.opt.weight_ramp:
                loss_val *= np.clip(self.opt.ramp_red*self.step / self.num_steps_per_epoch, 0.0, 1.0)

            losses[f'loss_coef/{loss_term}'] = loss_val

        for scale in self.opt.scales:

            losses_ps = {k:0 for k in loss_terms}     # defined per scale

            # Photometric Loss
            reprojection_losses = []
            color = inputs[('color', 0, scale)]
            target = inputs[('color', 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[('color', frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if self.bool_automask:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[('color', frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))
                # save both images, and do min all at once below
                identity_reprojection_loss = torch.cat(identity_reprojection_losses, 1)

            reprojection_loss = reprojection_losses

            if self.bool_automask:
                identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape, device=self.device) * 0.00001 # add random numbers to remove ties
                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if self.bool_automask:
                outputs['identity_selection/{}'.format(scale)] = (idxs > identity_reprojection_loss.shape[1] - 1).float()
            
            losses_ps['p_photo'] = to_optimise.mean()

            # Disparity Regularization 
            if move_Depth:
                if losses['loss_coef/d_smooth'] > 0:
                    disp = outputs[('disp', 0, scale)]
                    norm_disp = disp / (disp.mean(2, True).mean(3, True) + 1e-7)
                    losses_ps['d_smooth'] = compute_smooth_loss(norm_disp, color) / (2 ** scale)

                if losses['loss_coef/d_ground'] > 0 and self.base_model.bool_MotMask:
                    plane_dist, disp_diff, g_mask = self.process_ground(inputs, outputs, scale=scale)
                    disp_diff[disp_diff>0] = 0
                    losses_ps['d_ground'] = -1 * torch.mean(disp_diff) / (2 ** scale) # below ground is negative

            # Motion Regularization  

            num_frames = len(self.opt.frame_ids[1:])
            for frame_id in self.opt.frame_ids[1:]:

                disp = outputs[('disp', 0, scale)]                          # (B, 1, h, w) 
                color = inputs[('color', 0, scale)]                         # (B, 1, h, w) 
                motion_mask = outputs[('motion_mask', frame_id, scale)]     # (B, 1, h, w)
                h, w = motion_mask.shape[-2:]

                if move_CmpFlow and self.base_model.bool_CmpFlow:
                    complete_flow = outputs[('complete_flow', frame_id, scale)]     # (B, 3, h, w)
                    residual_flow = outputs[('residual_flow', frame_id, scale)]     # (B, 3, h, w)

                    if losses['loss_coef/c_smooth'] > 0:
                        losses_ps['c_smooth'] += compute_smooth_loss(complete_flow, color) / (2 ** scale) / num_frames

                    # consistency can only be computed when the motion mask is predicted as well
                    if self.base_model.bool_MotMask and losses['loss_coef/c_consistency'] > 0:
                        valid_disp = (disp > self.opt.mask_disp_thrd).detach()  # avoid rotational edge cases
                        losses_ps['c_consistency'] += torch.mean(valid_disp * (1-motion_mask.detach()) * torch.abs(residual_flow)) / (2 ** scale) / num_frames
                
                if move_MotMask and self.base_model.bool_MotMask:
                    sample_ego = outputs[('sample_ego', frame_id, scale)]               # (B, H, W, 2) 
                    sample_complete = outputs[('sample_complete', frame_id, scale)]     # (B, H, W, 2)
                    motion_prob = outputs[('motion_prob', frame_id, scale)]             # (B, 1, h, w)

                    if losses['loss_coef/m_sparsity'] > 0:
                        sample_ego = interp(sample_ego.permute(0, 3, 1, 2), (h, w))             # (B, 2, h, w) 
                        sample_complete = interp(sample_complete.permute(0, 3, 1, 2), (h, w))   # (B, 2, h, w) 
                        disp_mag = torch.sum((sample_ego - sample_complete) ** 2, 1)            # (B, h, w) 
                        static = (disp_mag < disp_mag.mean()).unsqueeze(1)                      # (B, 1, h, w) 
                        if torch.all(torch.sum(static, (1,2,3)) > 0):
                            losses_ps['m_sparsity'] += self.bce(motion_prob[static], self.prob_target[scale][static]) / (2 ** scale) / num_frames
                    
                    if losses['loss_coef/m_smooth'] > 0:
                        losses_ps['m_smooth'] += compute_smooth_loss(motion_mask, color) / (2 ** scale) / num_frames

            # Compile Losses
            for loss_term in loss_terms:
                losses[f'loss_term/{scale}'] += losses_ps[loss_term] * losses[f'loss_coef/{loss_term}']
                losses[f'loss_term/{loss_term}'] += losses_ps[loss_term]
            
            losses[f'loss'] += losses[f'loss_term/{scale}'] / self.num_scales

        return losses

    def compute_reprojection_loss(self, pred, target):
        """ Computes reprojection loss between a batch of predicted and target images
        """

        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = self.opt.ssim_weight * ssim_loss + (1-self.opt.ssim_weight) * l1_loss

        return reprojection_loss
    
    def process_ground(self, inputs, outputs, scale=0):
        """ Estimate and predict the ground plane given the disparity predictions
        """
        disp = outputs[('disp', 0, scale)] 
        _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        inv_K = inputs[('inv_K', scale)]
        H, W = self.H // (2**scale), self.W // (2**scale)

        cam_points = self.backproject_depth[scale](depth, inv_K)
        plane_dist, plane_param = self.gplane(cam_points[:,:3].reshape(-1,3,H,W))

        g_mask = (torch.abs(plane_dist) < self.opt.gp_tol).float()
        plane_param4diff = plane_param.clone()
        plane_param4diff[:, 2] += self.opt.gp_tol
        ground_disp, ground_depth  = self.get_ground_depth(plane_param4diff, inv_K, scale)

        disp_diff = disp - ground_disp
        disp_diff[ground_depth == self.opt.max_depth] = 0

        return plane_dist, disp_diff, g_mask
    
    def get_ground_depth(self, plane_param, inv_K, scale=0):
        """ Create a new disparity map that fills the holes indicated by the mask with the pixel below it
            :param plane_param  (B, 3, 1)
        """
        H, W = self.H // (2**scale), self.W // (2**scale)
        B = inv_K.size(0)
        cam_points_init = torch.matmul(inv_K[:, :3, :3], self.backproject_depth[scale].pix_coords[:B]) # (B, 3, H*W)

        w1, w2, w3 = plane_param[:,0:1], plane_param[:,1:2], plane_param[:,2:3]
        vx, vy, vz = cam_points_init[:,0:1], cam_points_init[:,1:2], cam_points_init[:,2:3]

        ground_depth = (w3 / (vy - vx*w1 - vz*w2)).reshape(B,1,H,W)
        ground_depth[torch.logical_or(ground_depth < 0, ground_depth > self.opt.max_depth)] = self.opt.max_depth
        ground_disp = depth_to_disp(ground_depth, min_depth=self.opt.min_depth, max_depth=self.opt.max_depth)
        
        return ground_disp, ground_depth
    

    # ========== Helper functions ========== #

    def setup_phase(self, phase_name):
        """ Setup proper base_model behaviors and self.optim
        """

        if phase_name == 'disp_init':
            self.base_model.bool_CmpFlow =   False
            self.base_model.bool_MotMask =     False
            self.optim = self.get_optim(['Depth', 'Pose'])

        elif phase_name == 'motion_init':
            self.base_model.bool_CmpFlow =   True
            self.base_model.bool_MotMask =     False
            self.optim = self.get_optim(['CmpFlow'])

        elif phase_name == 'mask_init':
            self.base_model.bool_CmpFlow =   True
            self.base_model.bool_MotMask =     True
            self.optim = self.get_optim(['Pose', 'CmpFlow', 'MotMask'])

        elif phase_name == 'fine_tune':
            self.base_model.bool_CmpFlow =   True
            self.base_model.bool_MotMask =     True
            self.optim = self.get_optim(['Depth', 'Pose', 'CmpFlow', 'MotMask'], lr_factor=0.5)
        else:
            raise Exception(f'Phase name {phase_name} not recognized.')

    def get_optim(self, network_names, optm=optim.Adam, lr_factor=1):
        """ construct optimizer and lr_scheduler based on the input module names
        """
        optimizer = optm(self.base_model.parameters_by_names(network_names), self.opt.learning_rate * lr_factor)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, self.opt.scheduler_step_size, 0.5)
        return {'optimizer': optimizer, 'lr_scheduler' : lr_scheduler, 'network_names' : network_names}

    def init_loaders(self):
        """ initialize self.train_loader and self.val_loader
        """
        self.setup_train_loader(verbose=True)
        self.setup_val_loader()

        self.num_steps_per_epoch = len(self.train_loader)
        self.val_iter = iter(self.val_loader)
        
        self.print('Number of training items / batches:    {} / {}'.format(len(self.train_dataset), len(self.train_loader)))
        self.print('Number of validation items / batches:  {} / {}\n'.format(len(self.val_dataset), len(self.val_loader)))

    def setup_train_loader(self, verbose=False):
        """ construct self.train_loader
        """
        train_filenames = readlines(osp.join(osp.dirname(__file__), 'splits', self.opt.split, 'train_files.txt'))

        if verbose:
            self.print(f'Total number of available training examples: {len(train_filenames)}')

        if self.opt.epoch_size > 0:
            total_batch_num = self.opt.batch_size * self.opt.local_world_size if self.opt.ddp else self.opt.batch_size
            num_example_per_epoch = total_batch_num * self.opt.epoch_size
            train_filenames = np.random.choice(train_filenames, num_example_per_epoch, replace=(num_example_per_epoch>len(train_filenames)))

        self.train_dataset = self.get_dataset(train_filenames, is_train=True, load_depth=False, load_mask=False)

        if self.opt.ddp:
            train_sampler, shf = DistributedSampler(self.train_dataset), False
        else:
            train_sampler, shf = None, True

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.B, shuffle=shf, num_workers=self.opt.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)


    def setup_val_loader(self):
        """ construct self.val_loader
        """

        # Note: using the same files as training - due to the unsupervised nature - validation is not used at all in the pipeline - only present for training monitoring
        # Note: ground truth depth is loaded but only used for training monitoring - ref: https://github.com/nianticlabs/monodepth2/blob/b676244e5a1ca55564eb5d16ab521a48f823af31/trainer.py#L499
        val_path = osp.join(osp.dirname(__file__), 'splits', self.opt.split, 'val_files.txt')
        train_path = osp.join(osp.dirname(__file__), 'splits', self.opt.split, 'train_files.txt')
        val_filenames = readlines(val_path) if osp.exists(val_path) else readlines(train_path)
        
        self.val_dataset = self.get_dataset(val_filenames, is_train=False, load_depth=True, load_mask=False)

        if self.opt.ddp:
            val_sampler, shf = DistributedSampler(self.val_dataset), False
        else:
            val_sampler, shf = None, True
        
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.B, shuffle=shf, num_workers=self.opt.num_workers, pin_memory=True, drop_last=True, sampler=val_sampler)


    def get_dataset(self, filenames, is_train=False, load_depth=False, load_mask=False, **kwargs):
        """ construct self.dataset 
        """
        dataset = self.dataset(data_path=self.opt.data_path, 
                               filenames=filenames, 
                               height=self.opt.height, 
                               width=self.opt.width,
                               cam_name=self.opt.cam_name,
                               img_type=self.opt.train_img_type,
                               frame_idxs=self.opt.frame_ids, 
                               num_scales=len(self.opt.scales), 
                               is_train=is_train, 
                               img_ext=self.opt.img_ext, 
                               load_depth=load_depth, 
                               load_mask=load_mask,
                               **kwargs)
        return dataset

    # ========== Log functions ========== #

    def vis_motion(self, depth, K, inv_K, motion_map=None, camTcam=None, scale=0):
        """ Compute optical flow map based on the input motion map and/or egomotion (camTcam)
            Projection via K and inv_K is used along with the depth predictions
        """

        assert motion_map != None or camTcam != None, 'At least one form of motion is supplied'
        b, _, h, w = depth.shape
        pix_ind_map = make_ind_map(h, w).to(self.device)

        # === obtain pix_motion_err from K and K_inv === # 
        cam_points = self.backproject_depth[scale](depth, inv_K)        # (B, 4, H*W)
        pix_coords, _ = self.project_3d[scale](cam_points, K, T=None)
        pix_motion_err = pix_coords - pix_ind_map                       # this should be zero - used for error correction
        
        # === compute raw optical flow === # 
        cam_points = self.backproject_depth[scale](depth, inv_K)        # (B, 4, H*W)
        if motion_map != None:
            b, _, h, w = motion_map.shape
            cam_points[:,:3,:] += motion_map.reshape(b, 3, h*w)
        pix_coords, _ = self.project_3d[scale](cam_points, K, camTcam)     # (B, H, W, 2)
        pix_motion_raw = pix_coords - pix_ind_map - pix_motion_err

        # === visualize optical flow === # 
        mag, theta = cart2polar(pix_motion_raw)
        max_mag = (mag.max().item() + 1e-8)
        hsv = torch.ones(b, 3, h, w).to(self.device)
        hsv[:, 0] = (theta - torch.pi/4) % (2 * torch.pi) / (2*torch.pi)
        hsv[:, 1] = 1.0
        hsv[:, 2] = mag / max_mag
        motion_visual = 1 - hsv_to_rgb(hsv)                             #(B, 3, H, W)

        return motion_visual, hsv, max_mag

    def log(self, mode, inputs, outputs, losses):
        """ Write an event to wandb
        """
        
        log_package = {f'{mode}_{k}' : v for k, v in losses.items()}
        if self.opt.no_train_vis:
            self.wandb_log(log_package)    # only log values
            return
            
        frame_id, s = -1, 0     # preset for consistent visualization
        
        # === First row: visualize rgb and reconstruction === # 
        color = inputs[('color', 0, 0)]                                 # rgb image
        recon_color = outputs[('color', frame_id, 0)]                   # reconstructed rgb image
        l1_raw = torch.abs(color-recon_color).mean(1, keepdim=True)
        l1_raw = l1_raw / (l1_raw.max() + 1e-6)                         # scaled l1 loss 

        # === Second row: visualize depth and motion mask === # 
        disp = outputs[('disp', 0, s)]                                  # disparity
        motion_mag = outputs[('motion_mask', frame_id, 0)]              # used exp mask as mag
        _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth) 
        
        # === Third row: visualize respective optical flows === # 
        motion = outputs[('independ_flow', frame_id, s)]                   # motion field
        K, inv_K, camTcam = inputs[('K', s)], inputs[('inv_K', s)], outputs[('cam_T_cam', 0, frame_id)]
        _, ego_hsv, ego_mag = self.vis_motion(depth, K, inv_K, motion_map=None, camTcam=camTcam, scale=s)   
        _, ind_hsv, ind_mag = self.vis_motion(depth, K, inv_K, motion_map=motion, camTcam=None, scale=s)        
        _, tot_hsv, tot_mag = self.vis_motion(depth, K, inv_K, motion_map=motion, camTcam=camTcam, scale=s)    
        max_mag = max(ind_mag, ego_mag, tot_mag)

        ego_hsv[:, 2] = torch.clamp(ego_hsv[:, 2] * ego_mag / max_mag, 0, 1)
        ind_hsv[:, 2] = torch.clamp(ind_hsv[:, 2] * ind_mag / max_mag, 0, 1)
        tot_hsv[:, 2] = torch.clamp(tot_hsv[:, 2] * tot_mag / max_mag, 0, 1)

        ego_flow = 1 - hsv_to_rgb(ego_hsv)
        ind_flow = 1 - hsv_to_rgb(ind_hsv)
        tot_flow = 1 - hsv_to_rgb(tot_hsv)

        # write images
        for j in range(self.B):
            first_row = torch.cat((color[j], recon_color[j], l1_raw[j].repeat(3,1,1)), 2)
            second_row = torch.cat((disp[j].repeat(3,1,1), motion_mag[j].repeat(3,1,1), depth[j].repeat(3,1,1) / torch.max(depth[j])), 2)
            third_row = torch.cat((ego_flow[j], ind_flow[j], tot_flow[j]), 2)

            out = torch.cat((first_row, second_row, third_row), 1)
            log_package[f'vis/{mode}_{j}'] = wandb.Image(out)
        
        self.wandb_log(log_package)
    
    def wandb_log(self, package):
        """ Wrapper to handle unexpected error with wandb
        """
        try:
            wandb.log(package, step=self.g_step)
        except:
            pass

    def log_time(self, batch_idx, duration, loss, data_time, gpu_time):
        """ Print a logging statement to the terminal
        """
        if not self.is_main():
            return

        samples_per_sec = self.B / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0

        print_string =  f'epoch {self.epoch:>3} | batch {batch_idx:>6} | examples/s: {samples_per_sec:5.1f} | loss: {loss:.5f} | '
        print_string += f'time elapsed: {sec_to_hm_str(time_sofar)} | time left: {sec_to_hm_str(training_time_left)} | CPU/GPU time: {data_time:0.1f}s/{gpu_time:0.1f}s'
        print(print_string)
    

    # ========== Load/Save functions ========== #

    def save_opt(self,):
        """ Save options to disk so we know what we ran this experiment with 
        """

        if not self.is_main():
            return

        models_dir = join_dir(self.log_path, 'models')
        to_save = self.opt.__dict__.copy()
        if self.opt.print_opt:
            for k,v in to_save.items():
                print('{:30}{}'.format(k+':', v))

        with open(osp.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, save_name='weights'):
        """ Save model weights and opt to disk
        """
        if not self.is_main():
            return

        save_folder = join_dir(self.log_path, 'models', f'{save_name}_{self.epoch:02}')
        self.base_model.save(save_folder)

        save_path = osp.join(save_folder, 'adam.pth')
        torch.save(self.optim['optimizer'].state_dict(), save_path)

    def load_model(self):
        """ Load model weights from disk 
        """
        self.base_model.load(verbose=self.is_main())       


    # ========== Utility functions ========== #

    def setup_wandb(self):
        """ initiate wandb server connection
        """
        wandb.init(project="Dynamo", name=self.opt.model_name, notes=self.opt.comment, config=self.opt)
    
    def process_inputs(self, inputs):
        """ resize inputs and add to self.device
        """
        self.apply_img_resize(inputs)
        for key, inp in inputs.items():
            inputs[key] = inp.to(self.device)
    
    def apply_img_resize(self, inputs):
        """ Apply resize operation in torch via self.resize and clamp values 
        """
        for scale in self.opt.scales:
            if scale != 0:
                inputs[('color', 0, scale)] = torch.clamp(self.resize[scale](inputs[('color', 0, scale-1)]), 0, 1)
        
    def is_main(self):
        """ check if self is the main trainer, always true if not 
        """
        return self.local_rank == 0

    def print(self, s=""):
        """ print string s if self is main
        """
        if self.is_main():
            print(s)

    def set_train(self):
        """ Convert all models to training mode
        """
        self.base_model.set_train()

    def set_eval(self):
        """ Convert all models to testing/evaluation mode 
        """
        self.base_model.set_eval()
    
