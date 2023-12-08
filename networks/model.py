import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import transformation_from_parameters
from .resnet_encoder import ResnetEncoder
from .depth_encoder import LiteMono
from .depth_decoder import DepthDecoder
from .depth_decoder import LiteDepthDecoder
from .pose_decoder import PoseDecoder
from .motion_decoder import MotionDecoder

class Model(nn.Module):

    def __init__(self, options):
        super(Model, self).__init__()
        self.opt = options

        if self.opt.depth_model == 'monodepthv2':
            self.depth_enc = ResnetEncoder(self.opt.encoder_num_layers, self.opt.weights_init == 'pretrained')
            self.depth_dec = DepthDecoder(self.depth_enc.num_ch_enc, self.opt.scales)
        elif self.opt.depth_model == 'litemono':
            self.depth_enc = LiteMono(model='lite-mono-8m', drop_path_rate=0.4, pretrained=self.opt.weights_init == 'pretrained')
            self.depth_dec = LiteDepthDecoder(self.depth_enc.num_ch_enc, self.opt.scales)
        else:
            raise Exception(f'Model Name {self.opt.depth_model} not recognized.')  
            
        self.pose_enc = ResnetEncoder(self.opt.encoder_num_layers, self.opt.weights_init == 'pretrained', num_input_images=2, inp_disp=False)
        self.pose_dec = PoseDecoder(self.pose_enc.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
        self.motion_enc = ResnetEncoder(self.opt.encoder_num_layers, self.opt.weights_init == 'pretrained', num_input_images=3, inp_disp=False)
        self.motion_dec = MotionDecoder(self.pose_enc.num_ch_enc, self.opt.scales, num_input_images=3, inp_disp=False, out_dim=3)    # full motion decoder
        self.motion_mask = MotionDecoder(self.pose_enc.num_ch_enc, self.opt.scales, num_input_images=3, inp_disp=False, out_dim=1)    # probability that is independently moving
        
        self.network2modules = {
            'Depth'  : ['depth_enc', 'depth_dec'],   # Depth Network, D
            'Pose'   : ['pose_enc', 'pose_dec'],     # Pose Network, P
            'CmpFlow': ['motion_enc', 'motion_dec'], # Complete Flow Network, C (share encoder)
            'MotMask': ['motion_enc', 'motion_mask'] # Motion Mask Network, M   (share encoder)
        }

        self.module_names = list(set([mod for mods in self.network2modules.values() for mod in mods]))

        self.bool_CmpFlow = True    # predict complete 3d motion flow
        self.bool_MotMask = True    # predict binary motion mask

        # google drive links
        self.model_zoo = {
                "ckpt/K_Dynamo-Depth_MD2" : '1SLQcCQplfAtqeWUD4TQc42aGpevViTGX',
                "ckpt/K_Dynamo-Depth" : '1b1kwxqUquFbSMU9WLAr6_pIbj1HxoWLJ',
                "ckpt/N_Dynamo-Depth_MD2" : '1t0Z_2hD0raAi4vDK_VZFXIcwcTFx0elU',
                "ckpt/N_Dynamo-Depth" : '1oqQVFyGxo_SxclpinrBlwGSE1gEfVAZY',
                "ckpt/W_Dynamo-Depth_MD2" : None,   # please reach out according to the README
                "ckpt/W_Dynamo-Depth" : None        # please reach out according to the README
            }

    def forward(self, inputs):

        outputs = dict()
        self.predict_depths(inputs, outputs)
        self.predict_poses(inputs, outputs)
        self.predict_motions(inputs, outputs)

        return outputs
    
    # ========== Prediction functions ========== #

    def predict_depths(self, inputs, outputs):
        for f_i in self.opt.frame_ids:
            depth_features = self.depth_enc(inputs['color_aug', f_i, 0])
            depth_out = self.depth_dec(depth_features)

            outputs.update({(k[0], f_i, k[1]) : v for k,v in depth_out.items()})

    def predict_poses(self, inputs, outputs):
        """ Predict egomotion (in a form of rotation and translation)
        """

        pose_inputs = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

        for f_i in self.opt.frame_ids[1:]:

            # Ignoring order, always pass in target frame last
            pose_input = torch.cat([pose_inputs[f_i], pose_inputs[0]], 1)
            pose_feats = [self.pose_enc(pose_input)]

            axisangle, translation = self.pose_dec(pose_feats)
            axisangle, translation = axisangle[:, 0], translation[:, 0]

            outputs.update({
                ('pose_feats', 0, f_i) :   [pose_input] + pose_feats[0],
                ('axisangle', 0, f_i)   :   axisangle,
                ('translation', 0, f_i) :   translation,
                ('cam_T_cam', 0, f_i)   :   transformation_from_parameters(axisangle, translation, invert=True),
            })
    
    def predict_motion_feat(self, inputs, outputs):
        """ Predict egomotion (in a form of rotation and translation)
        """

        motion_inputs = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

        for f_gap in set([abs(f_i) for f_i in self.opt.frame_ids[1:]]):
            f_prev, f_next = -1 * f_gap, f_gap

            # Keep order and center frame at zero
            motion_input = torch.cat([motion_inputs[f_prev], motion_inputs[0], motion_inputs[f_next]], 1)
            motion_feats = self.motion_enc(motion_input)

            outputs.update({
                ('motion_feats', 0, f_gap) :   [motion_input] + motion_feats,
            })

    
    def predict_motions(self, inputs, outputs):
        """ Predict independent object motion 
        """

        if not self.bool_CmpFlow and not self.bool_MotMask:
            return  # early terminate since no need for output

        self.predict_motion_feat(inputs, outputs)

        for f_gap in set([abs(f_i) for f_i in self.opt.frame_ids[1:]]):
            f_prev, f_next = -1 * f_gap, f_gap

            motion_input = outputs[('motion_feats', 0, f_gap)]

            # subtraction since order is ignored during, obtain its mean
            ego_translation = (outputs[('translation', 0, f_prev)].detach() - outputs[('translation', 0, f_next)].detach()) / 2
            ego_axisangle = (outputs[('axisangle', 0, f_prev)].detach() - outputs[('axisangle', 0, f_next)].detach()) / 2
            ego_motion = torch.cat((ego_translation, ego_axisangle), -1).permute(0,2,1).unsqueeze(3)

            if self.bool_CmpFlow:
                motion_out = self.motion_dec(motion_input, ego_motion)

                # full motion predictions need to be inverted for finding a point back in time
                # always using frame 0 as reference, so it is omitted -> (name, f_i, scale)
                outputs.update({(k[0], f_prev, k[1]) : -1 * v for k,v in motion_out.items()}) 
                outputs.update({(k[0], f_next, k[1]) :  1 * v for k,v in motion_out.items()}) 
            
            if self.bool_MotMask:
                motion_prob = self.motion_mask(motion_input, ego_motion)

                # motion probabilities just need to be duplicated
                # always using frame 0 as reference, so it is omitted -> (name, f_i, scale)
                outputs.update({(k[0], f_prev, k[1]) : v for k,v in motion_prob.items()})
                outputs.update({(k[0], f_next, k[1]) : v for k,v in motion_prob.items()})


    # ========== Helper functions ========== #
    
    def parameters_by_names(self, network_names):
        """ Returns the list of parameters associated with the given network_names
        """
        parameters_to_train = []
        module_names = list(set([mod for network in network_names for mod in self.network2modules[network]]))
        for module_name in module_names:
            parameters_to_train += list(getattr(self, module_name).parameters())
        return parameters_to_train
    
    def save(self, save_folder):
        """ Save model weights to disk
        """
        for module_name in self.module_names:
            save_path = osp.join(save_folder, '{}.pth'.format(module_name))
            to_save = getattr(self, module_name).state_dict()
            if 'enc' in module_name:
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            torch.save(to_save, save_path)
    
    def load(self, dev='cpu', verbose=True):
        """ Load model(s) from disk 
        """
        self.opt.load_ckpt = osp.expanduser(self.opt.load_ckpt)
        self.check_load_ckpt(self.opt.load_ckpt)

        if verbose:
            print('loading model from folder {}'.format(self.opt.load_ckpt))

        for module_name in self.module_names:
            path = osp.join(self.opt.load_ckpt, f'{module_name}.pth')
            if not osp.exists(path):
                if verbose:
                    print(f'|- Loading {module_name} weights... FAILED :: Path {path} not found')
                continue

            checkpoint = torch.load(path, map_location=dev)

            if 'height' in checkpoint:
                if not (checkpoint['height'] == self.opt.height and checkpoint['width'] == self.opt.width):
                    if verbose:
                        print('|- === WARNING: self.opt ({},{}) != loaded ({},{})'.format(self.opt.height, self.opt.width, checkpoint['height'], checkpoint['width']))
                checkpoint.pop('height')
                checkpoint.pop('width')
            try:
                if verbose:
                    print(f'|- Loading {module_name} weights...')
                getattr(self, module_name).load_state_dict(checkpoint)
            except:
                if verbose:
                    print(f'|- Loading {module_name} weights... FAILED :: load_state_dict() mismatch - Loading Matched Parameters.' )

                model_dict = getattr(self, module_name).state_dict() 
                model_dict.update({k: v for k, v in checkpoint.items() if k in model_dict})
                getattr(self, module_name).load_state_dict(model_dict)

    def check_load_ckpt(self, load_ckpt):
        """ Check if the load ckeckpoint is present or downloadable
        """
        if not osp.isdir(load_ckpt):
            if load_ckpt in self.model_zoo:
                print(f'Missing model checkpoint {load_ckpt}, downloading it now.')
                model_name = load_ckpt.split('/')[1]
                os.makedirs('./ckpt/', exist_ok=True)
                if self.model_zoo[load_ckpt] is None:
                    raise Exception(f'Cannot load ckpt {load_ckpt} due to waymo license, please reach out for access according to the README')
                os.system(f'gdown {self.model_zoo[load_ckpt]}; unzip {model_name}.zip; mv {model_name} ckpt/{model_name}; rm {model_name}.zip')
            else:
                raise Exception('Cannot find folder {}'.format(self.opt.load_ckpt))

    def set_train(self):
        for module_name in self.module_names:
            getattr(self, module_name).train()
    
    def set_eval(self):
        for module_name in self.module_names:
            getattr(self, module_name).eval()
