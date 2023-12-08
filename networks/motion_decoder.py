import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

class MotionDecoder(nn.Module):

    def __init__(self, num_inp_feat, scales=4, num_input_images=2, inp_disp=True, out_dim=4):
        super(MotionDecoder, self).__init__()

        self.org_in_ch = num_input_images * (3 + int(inp_disp))
        self.num_inp_feat = num_inp_feat[::-1].tolist() + [self.org_in_ch]
        self.out_dim = out_dim
        self.scales = scales
        self.softmax = nn.Softmax(dim=1)

        assert max(self.scales) < len(self.num_inp_feat)

        self._residual_translation = nn.Conv2d(6, self.out_dim, kernel_size=1, stride=1, padding=0)

        self.refine_convs = []
        
        for ii, enc_in_dim in enumerate(self.num_inp_feat):
            setattr(self, f'refine_motion_conv{ii}', self._refine_motion_conv(enc_in_dim))
            setattr(self, f'refine_motion_redu{ii}', self._refine_motion_redu(enc_in_dim))
    
    def _refine_motion_conv(self, in_dim):
        return nn.Sequential(nn.Conv2d(in_dim + self.out_dim, in_dim, kernel_size=3, stride=1, padding=1),
                             nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1))

    def _refine_motion_redu(self, in_dim):
        return nn.Conv2d(in_dim*2, self.out_dim, kernel_size=1, stride=1)

    def _upsample_and_concat(self, motion_field, ii):
        out_feat = self.outputs[-1-ii]
        b, c, h, w = out_feat.shape

        upsampled_motion_field = F.interpolate(motion_field, size=(h, w), mode='bilinear', align_corners=False)
        conv_input = torch.cat((upsampled_motion_field, out_feat.to(device=motion_field.device)), 1)

        return conv_input, upsampled_motion_field
    
    def padding(self, x):
        in_height, in_width = x.shape[2], x.shape[3]
        out_height = ceil(float(in_height) / float(1))
        out_width = ceil(float(in_width) / float(1))

    def _refine_motion_field(self, x):

        out = {'x_-1_out' : x}

        for ii in range(len(self.num_inp_feat)):
            out[f'x_{ii}'], out[f'upsample_motion_{ii}'] = self._upsample_and_concat(out[f'x_{ii-1}_out'], ii)      # (B, num_inp_feat[ii]+out_dim, h, w), (B, out_dim, h, w) 
            
            conv, redu = getattr(self, f'refine_motion_conv{ii}'), getattr(self, f'refine_motion_redu{ii}')

            out[f'x_{ii}_1'] =        conv[:-1](out[f'x_{ii}'])                                                 # (B, num_inp_feat[ii], h, w)
            out[f'x_{ii}_2'] =        conv[-1](out[f'x_{ii}_1'])                                                # (B, num_inp_feat[ii], h, w)
            out[f'x_{ii}_reduced'] =  redu(torch.cat((out[f'x_{ii}_1'], out[f'x_{ii}_2']), 1))                  # (B, out_dim, h, w)
            out[f'x_{ii}_out'] =      out[f'x_{ii}_reduced'] + out[f'upsample_motion_{ii}']                     # (B, out_dim, h, w)
            
        return out


    def forward(self, pose_feat, ego_motion):
        '''
        pose_feat: [(B, 6, H, W), (B, 64, H//2, W//2), (B, 64, H//4, W//4), (B, 128, H//8, W//8), (B, 256, H//16, W//16), (B, 512, H//32, W//32)]
        disp_feat: [(B, 2, H, W), (B, 2, H//2, W//2), (B, 2, H//4, W//4), (B, 2, H//8, W//8), (B, 2, H//16, W//16), (B, 2, H//32, W//32)]
        ego_motion: (B, 6, 1, 1)
        '''

        outputs = dict()
        self.outputs = pose_feat
 
        res_trans = self._residual_translation(100 * ego_motion)    # (B, out_dim, 1, 1)
        out = self._refine_motion_field(res_trans)                  # (B, out_dim, H, W)

        for scale in self.scales:
            ii = len(self.num_inp_feat) - 1 - scale 
            res_trans = out[f'x_{ii}_out']
            m_raw =     0.01 * res_trans

            if self.out_dim == 1:       # used to predict binary motion mask
                outputs[('motion_prob', scale)] =    m_raw                          # (B, 1, H, W) 
                outputs[('motion_mask', scale)] =    torch.sigmoid(m_raw)           # (B, 1, H, W) 
            elif self.out_dim == 3:     #  used to predict complete 3D flow
                outputs[('complete_flow', scale)] =    m_raw          # (B, 3, H, W)
            else:
                raise Exception(f'out_dim={self.out_dim} not excepted.')

        return outputs


