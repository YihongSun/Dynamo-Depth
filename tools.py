import torch
import torch.nn as nn
import numpy as np


class DepthMetrics(nn.Module):
    """ Compute depth performance
    """
    def __init__(self, img_bound, min_depth, max_depth):
        super(DepthMetrics, self).__init__()
        self.depth_metric_names = ['de:abs_rel', 'de:sq_rel', 'de:rms', 'de:log_rms', 'da:a1', 'da:a2', 'da:a3']
        self.img_bound = img_bound
        self.min_depth = min_depth
        self.max_depth = max_depth
    
    def forward(self, inputs, outputs, mask=None):
        disp_pred = outputs[('disp_scaled', 0, 0)]  # (B, 1, H, W)
        depth_gt = inputs['depth_gt']               # (B, dataset.max_depth_samp, 3)    # include padding
        depth_valid = inputs['depth_valid']         # (B, dataset.max_depth_samp,)      # specify original
        gt_dim = inputs['gt_dim']                   # (B, 2,) - ground truth image dim

        metrics = {metric : 0 for metric in self.depth_metric_names}
        if mask is not None:
            mask_labels = [l.item() for l in torch.unique(mask)] # split based on mask values
            metrics.update({f'{metric}_mask' : {l : [0,0] for l in mask_labels} for metric in self.depth_metric_names})

        for bi, (disp_p, depth_g, valid, dim) in enumerate(zip(disp_pred, depth_gt, depth_valid, gt_dim)):

            gt_height, gt_width = dim[0].item(), dim[1].item()
            up, down = int(self.img_bound[0] * gt_height), int(self.img_bound[1] * gt_height)
            left, right = int(self.img_bound[2] * gt_width), int(self.img_bound[3] * gt_width)

            valid = torch_and(valid,
                              depth_g[:,0] >= up,                     # check for image bounds
                              depth_g[:,0] < down,
                              valid, depth_g[:,1] >= left,
                              valid, depth_g[:,1] < right,
                              depth_g[:,2] > self.min_depth, # check for depth bounds
                              depth_g[:,2] < self.max_depth)
            
            valid_ind = depth_g[:,0][valid].long(), depth_g[:,1][valid].long()
            depth_p = 1 / nn.functional.interpolate(disp_p[None], (gt_height, gt_width), mode='bilinear', align_corners=False).squeeze()

            d_gt = depth_g[:,2][valid]
            d_pd = depth_p[valid_ind]

            # median scaling and clamp
            d_pd *= torch.median(d_gt) / torch.median(d_pd)
            d_pd = torch.clamp(d_pd, self.min_depth, self.max_depth)

            depth_errors = compute_errors(d_gt, d_pd)
            for i, metric in enumerate(self.depth_metric_names):
                metrics[metric] += depth_errors[i]

            if mask is not None:
                m_valid = mask[bi][valid_ind]
                for l in mask_labels:
                    m = m_valid == l

                    dgm, dpm = d_gt[m], d_pd[m]
                    cnt = dgm.shape[0]
                    if cnt == 0:
                        continue
                    depth_errors = compute_errors(dgm, dpm)
                    
                    for i, metric in enumerate(self.depth_metric_names):
                        metrics[f'{metric}_mask'][l][0] += depth_errors[i].item() * cnt  
                        metrics[f'{metric}_mask'][l][1] += cnt  
        
        for metric in self.depth_metric_names:
            metrics[metric] = metrics[metric] / disp_pred.size(0)
            
        return metrics


class GroundPlane(nn.Module):
    def __init__(self, num_points_per_it=5, max_it=25, tol=0.1, g_prior=0.5, vertical_axis=1):
        super(GroundPlane, self).__init__()
        self.num_points_per_it = num_points_per_it
        self.max_it = max_it
        self.tol = tol
        self.g_prior = g_prior
        self.vertical_axis = vertical_axis
    
    def forward(self, points):
        """ estiamtes plane parameters and return each points distance to it
        :param points     (B, 3, H, W)
        :ret distance     (B, 1, H, W)
        :ret plane_param  (B, 3)
        """
        
        B, _, H, W = points.shape 
        ground_points = points[:, :, -int(self.g_prior*H):, :]
        ground_points_inp = ground_points.reshape(B, 3, -1).permute(0,2,1) # (B, N, 3)

        plane_param = self.estimate_ground_plane(ground_points_inp)
        
        all_points = points.reshape(B, 3, H*W).permute(0,2,1)  # (B, H*W, 3)
        dist2plane = self.dist_from_plane(all_points, plane_param).permute(0,2,1).reshape(B,1,H,W)

        return dist2plane.detach(), plane_param.detach()
    
    def dist_from_plane(self, points, param):
        """ get vertical distance of each point from plane specified by param
        :param points   (B, 3) or (SB, B, 3)
        :param param    (3, 1) or (SB, 3, 1)
        :ret distance   (B, 1) or (SB, B, 1)
        """
        
        A, B = self.get_AB(points)
        return A @ param - B
        

    def estimate_ground_plane(self, points):
        """
        :param points           (B, N, 3)            
        :ret plane parameter    (B, 3) (B)
        """
        
        B, N, _ = points.shape
        T = self.num_points_per_it*self.max_it

        rand_points = []

        for b in range(B):
            rand_ind = np.random.choice(np.arange(N), T, replace=True)
            rand_points.append(points[b][rand_ind])
        rand_points = torch.stack(rand_points)  # (B, T, 3)
        
        ws = self.calc_param(rand_points).reshape(-1, 3, 1)    # (B*self.max_it, 3, 1)
        ps = points.repeat(self.max_it,1,1)                    # (B*self.max_it, N, 3)
        
        abs_dist = torch.abs(self.dist_from_plane(ps, ws)).reshape(B, self.max_it, N)

        param_fit = (abs_dist < self.tol).float().mean(2)
        best_fit = param_fit.argmax(1)
        best_w = ws.reshape(B, self.max_it, 3, 1)[np.arange(B),best_fit]

        return best_w
        
    def calc_param(self, points):
        """
        :param points           (B, self.max_it, self.num_points_per_it, 3)            
        :ret plane parameter    (B, self.max_it, 3)
        """
            
        batched_points = points.reshape(-1, self.num_points_per_it, 3)
        
        A, B = self.get_AB(batched_points)
        At = A.transpose(2,1) # batched transpose
        
        w = (torch.inverse(At @ A + 1e-6) @ At @ B).reshape(points.size(0), self.max_it, 3, 1)

        return w
    
    def get_AB(self, points):
        """ get mat A and B associated with points
        :param points   (B, 3) or (SB, B, 3)
        :ret A    (B, 3) or (SB, B, 3)
        :ret B    (B, 1) or (SB, B, 1)
        """
        B = points[..., self.vertical_axis:self.vertical_axis+1]
        A = torch.cat([points[..., i:i+1] for i in range(3) if i != self.vertical_axis] + [torch.ones_like(B)], -1)
        return A, B


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        B = depth.size(0)
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords[:B])
        cam_points = depth.view(B, 1, -1) * cam_points[:B]
        cam_points = torch.cat([cam_points, self.ones[:B]], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        # P = torch.matmul(K, T)[:, :3, :]
        cam_points_3D = torch.matmul(T, points) if T is not None else points
        cam_points = torch.matmul(K[:, :3, :], cam_points_3D)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)   #pinhole model - normalize by plane in front, not a spherical ball
        pix_coords = pix_coords.view(-1, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        ego_motion = cam_points_3D[:,:3] - points[:,:3]

        return pix_coords, ego_motion


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

def torch_and(*args):
    """ Accept a list of arugments of torch.Tensor of the same shape, compute element-wise and operation for all of them
        Output tensor has the same shape as the input tensors
    """
    out = args[0]
    for a in args:
        assert out.size() == a.size(), "Sizes must match: [{}]".format(', '.join([str(x.size()) for x in args]))
        out = torch.logical_and(out, a)
    return out

def compute_errors(gt, pred):
    """ Computation of error metrics between predicted and ground truth depths
        https://github.com/nianticlabs/monodepth2/blob/b676244e5a1ca55564eb5d16ab521a48f823af31/evaluate_depth.py#L27
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def depth_to_disp(depth, min_depth, max_depth):
    """Inverse of the previous function
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = 1 / depth
    disp = (scaled_disp - min_disp) / (max_disp - min_disp)
    return disp


def compute_smooth_loss(inp, img=None):
    """ Computes the smoothness loss for an arbitrary tensor of size [B, C, H, W]
        The color image is used for edge-aware smoothness
    """

    grad_inp_x = torch.abs(inp[:, :, :, :-1] - inp[:, :, :, 1:])
    grad_inp_y = torch.abs(inp[:, :, :-1, :] - inp[:, :, 1:, :])

    if img is not None:
        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_inp_x *= torch.exp(-grad_img_x)
        grad_inp_y *= torch.exp(-grad_img_y)

    return grad_inp_x.mean() + grad_inp_y.mean()

