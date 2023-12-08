import os, cv2
import os.path as osp
import numpy as np
import imageio
import torch
import matplotlib as mpl
import matplotlib.cm as cm

def readlines(filename):
    """ Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def write_to_file(data_list, fname, bool_newline=True):
    """ Write the given list of strings into the file
    """
    with open(fname, 'w') as fh:
        if bool_newline:
            fh.writelines([d+'\n' for d in data_list])
        else:
            fh.writelines(data_list)

def get_model_ckpt_name(load_path):
    """ Parse model name and checkpoint name from the load path, used by eval/*.py
        :param load_path - path of the loaded ckpt
        :return model_name, ckpt_name
    """
    load_path_list = load_path.split('/') 

    # since loaded paths are commonly found under logs/ or ckpt/ -> more intelligent parsing for these two cases
    if 'logs' in load_path_list:   
        # [..., 'logs', model_name, 'models', ckpt_name, ...]
        model_name = load_path_list[load_path_list.index('logs')+1]
        ckpt_name = load_path_list[load_path_list.index('logs')+3]

    elif 'ckpt' in load_path_list: 
        # [..., 'ckpt', model_name, ...]
        model_name = load_path_list[load_path_list.index('ckpt')+1]
        ckpt_name = 'ckpt'

    else:
        model_name = '[{}]'.format('-'.join(load_path_list))
        ckpt_name = 'ckpt'
        print(f'Loaded path (={load_path}) does not appear to be under logs/ or ckpt/')
        print(f'\tUsing general model_name=`{model_name}` and ckpt_name=`{ckpt_name}`.')
    
    return model_name, ckpt_name

def get_filenames(segment_name, opt):
    """ Return the list of filenames given a segment path
    """
    cam_name, img_type, img_ext = opt.cam_name, opt.eval_img_type, opt.eval_img_ext
    rgb_dir_path = osp.join(opt.data_path, segment_name, cam_name, 'rgb', img_type)
    frame_indices = sorted([int(osp.splitext(f)[0]) for f in os.listdir(rgb_dir_path) if osp.splitext(f)[1] == img_ext])
    return [f'{segment_name} {i}' for i in frame_indices]

def is_edge(filename, opt):
    """ Determine if the given filename is on the edge of the sequence given the range of opt.frame_ids
        Only used during evaluation
    """
    cam_name, img_type, img_ext = opt.cam_name, opt.eval_img_type, opt.eval_img_ext
    seg_name, frame_index = filename.split()[0], int(filename.split()[1])
    left_index, right_index = frame_index + np.min(opt.frame_ids), frame_index + np.max(opt.frame_ids)
    left_bound = osp.join(opt.data_path, seg_name, cam_name, 'rgb', img_type, f'{left_index:06}{img_ext}')
    right_bound = osp.join(opt.data_path, seg_name, cam_name, 'rgb', img_type, f'{right_index:06}{img_ext}')
    return (not osp.exists(left_bound)) or (not osp.exists(right_bound))

def join_dir(*tree):
    """ Return joined path and create directory in every level if doesn't exist
    """
    path = osp.join(*tree)
    if not osp.exists(path):
        try:
            os.makedirs(path, exist_ok=True)
        except:
            pass # catch sync racing errors
    return path

def make_mp4(images, filename, fps=30, quality=8, macro_block_size=1, bgr=True):
    """ Saves the images as video with given fps, quality and macro_block_size
        Assumes the image images uses bgr format (loaded from cv2 instead of PIL)
    """
    file_ext = osp.splitext(filename)[1]

    if file_ext == "":
        filename = filename + '.mp4'
    elif file_ext != ".mp4":
        raise Exception(f'Given filename does not end with .mp4 : filename=`{filename}`')

    frames = np.stack(images, axis=0)
    if bgr: # bgr -> rgb
        frames = frames[...,::-1] 

    imageio.mimwrite(filename, frames, fps=fps, quality=quality, macro_block_size=macro_block_size)

def interp(x, shape, mode='bilinear', align_corners=False):
    """ Image tensor interpolation of x with shape (B, C, H, W) -> (B, C, *shape)
    """
    return torch.nn.functional.interpolate(x, shape, mode=mode, align_corners=align_corners)

def score_map_vis(score_map, cmap='bone', vminmax=None, max_perc=95):
    """ Accepts score_map as torch.Tensor of shape [1, 1, h, w] or np.ndarray of shape [h, w]
        Assumes the image images uses bgr format (loaded from cv2 instead of PIL)
    """
    score_map_np = score_map.squeeze().cpu().numpy() if torch.is_tensor(score_map) else score_map   # either torch.Tensor or np.ndarray

    if vminmax == None:
        vmin = score_map_np.min()
        vmax = np.percentile(score_map_np, max_perc)
    else:
        vmin, vmax = vminmax

    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    return mapper.to_rgba(score_map_np)[:, :, :3]

def make_ind_map(height, width):
    """ Create identity indices map of shape (1,H,W,2) where top left corner is [-1,-1] and bottom right corner is [1,1]
    """
    v_strip = torch.arange(0, height) / height * 2 - 1
    h_strip = torch.arange(0, width) / width * 2- 1

    return torch.stack([h_strip.unsqueeze(0).repeat(height, 1), v_strip.unsqueeze(1).repeat(1, width),]).permute(1,2,0).unsqueeze(0)

def cart2polar(cart):
    """ Convert cartian points into polar coordinates, the last dimension of the input must contain y and x component
    """
    assert cart.shape[-1] == 2, 'Last dimension must contain y and x vector component'

    r = torch.sqrt(torch.sum(cart**2, -1))
    theta = torch.atan(cart[...,0]/cart[...,1])
    theta[torch.where(torch.isnan(theta))] = 0  # torch.atan(0/0) gives nan

    theta[cart[...,1] < 0] += torch.pi
    theta = (5*torch.pi/2 - theta) % (2*torch.pi)

    return r, theta

def hsv_to_rgb(image):
    """ Convert image from hsv to rgb color space, input must be torch.Tensor of shape (*, 3, H, W)
    """
    assert isinstance(image, torch.Tensor), f"Input type is not a torch.Tensor. Got {type(image)}"
    assert len(image.shape) >= 3 and image.shape[-3] == 3, f"Input size must have a shape of (*, 3, H, W). Got {image.shape}"

    h = image[..., 0, :, :]
    s = image[..., 1, :, :]
    v = image[..., 2, :, :]
    

    hi = torch.floor(h * 6) % 6
    f = ((h * 6) % 6) - hi
    one = torch.tensor(1.0, device=image.device, dtype=image.dtype)
    p = v * (one - s)
    q = v * (one - f * s)
    t = v * (one - (one - f) * s)

    hi = hi.long()  # turns very negative for nan
    indices = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    out = torch.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q), dim=-3)
    out = torch.gather(out, -3, indices)

    return out

def sec_to_hm(t):
    """ Convert time in seconds to time in hours, minutes and seconds 
        e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s

def sec_to_hm_str(t):
    """ Convert time in seconds to a nice string
        e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)
