import os, argparse
file_dir = os.path.dirname(__file__)

class DynamoOptions:
	def __init__(self):
		self.p = argparse.ArgumentParser(description="Dynamo options")
		# NOTE: For arguments that are dataset-dependent, `None` is used for default as placeholders. Refer to `parse(self, **kwargs)` for the actual default values

		# EXPERIMENT options
		self.p.add_argument("--model_name", "-n",
							type=str,
							help="the name of the folder to save the model in",
							default="--")
		self.p.add_argument("--log_dir",
							type=str,
							help="log directory",
							default="./logs")
		self.p.add_argument("--eval_dir",
							type=str,
							help="evalutation directory",
							default="./outputs")
		

		# SYSTEM options
		self.p.add_argument("--cuda_ids",
							nargs="+",
							type=int,
							help="cuda device ids to use - use ddp if len() > 1 and cuda_ids[0] for eval/vis",
							default=[0])
		self.p.add_argument("--local_rank",
							type=int,
							help="local rank of the trainer",
							default=0)
		self.p.add_argument("--ddp",
							type=bool,
							help="boolean: always false, only set True by train.py",
							default=False)
		self.p.add_argument("--num_workers",
							type=int,
							help="number of dataloader workers",
							default=2)
		

		# DATASET options
		self.p.add_argument("--dataset", "-d",
							type=str,
							help="dataset to train on",
							choices=["kitti", "waymo", "nuscenes"],
							default="waymo")
		self.p.add_argument("--data_path",
							type=str,
							help="path to the training data",
							default=None)	
		self.p.add_argument("--split",
							type=str,
							help="which train/val split to use",
							default=None)
		self.p.add_argument("--height",
							type=int,
							help="input image height",
							default=None)
		self.p.add_argument("--width",
							type=int,
							help="input image width",
							default=None)
		self.p.add_argument("--img_ext",
							type=str,
							help="extension of images to be loaded",
							choices=[".png", ".jpg"],
							default=".jpg")
		self.p.add_argument("--cam_name",
							type=str,
							help="which camera to use",
							default=None)


		# LOSS weights
		self.p.add_argument("--g_p_photo",
							type=float,
							help="photometric ssim weight gamma against l1 loss",
							default=1.0)
		self.p.add_argument("--g_d_smooth",
							type=float,
							help="disparity smoothness weight gamma",
							default=1e-3)
		self.p.add_argument("--g_d_ground",
							type=float,
							help="disparity above ground assumption weight gamma",
							default=0.1)
		self.p.add_argument("--g_c_smooth",
							type=float,
							help="complete 3d flow smoothness weight gamma",
							default=1e-3)	
		self.p.add_argument("--g_c_consistency",
							type=float,
							help="complete 3d flow consistency with ego 3d flow at static regions (require mask prediction) weight gamma",
							default=5.0)
		self.p.add_argument("--g_m_sparsity",
							type=float,
							help="motion mask sparsity weight gamma",
							default=0.04)
		self.p.add_argument("--g_m_smooth",
							type=float,
							help="motion mask smoothness weight gamma",
							default=0.1)
		self.p.add_argument("--weight_ramp",
							nargs="+",
							type=str,
							help="loss coefficients that requires weight ramp",
							default=['g_c_smooth', 'g_c_consistency', 'g_m_sparsity', 'g_m_smooth'])
		self.p.add_argument("--ramp_red",
							type=float,
							help="factor in which the weight ramp is reduced by",
							default=3)
		self.p.add_argument("--ssim_weight",
							type=float,
							help="photometric ssim weight gamma against l1 loss",
							default=0.85)
		self.p.add_argument("--mask_disp_thrd",
							type=float,
							help="disparity threshold for motion mask ignore when computing complete flow consistency loss",
							default=0.03)			
		

		# TRAINING hyperparameters
		self.p.add_argument("--epoch_schedules",
							nargs="+",
							type=int,
							help="[disp_init, motion_init, mask_init, fine_tune]\
								disp_init=	# epochs used for init. disparity D + ego-motion P\
								motion_init=# epochs used for init. full motion C\
								mask_init=	# epochs used for init. motion mask M\
								fine_tune= 	# epochs used for E2E fine-tuning]",
							default=[1, 1, 5, 20])
		self.p.add_argument('--epoch-size', 
							type=int, 
							help='manual epoch size (will match dataset size if 0)',
							default=8000)
		self.p.add_argument("--batch_size", "-b",
							type=int,
							help="batch size",
							default=3)
		self.p.add_argument("--learning_rate",
							type=float,
							help="learning rate",
							default=1e-4)
		self.p.add_argument("--scheduler_step_size",
							type=int,
							help="step size of the scheduler",
							default=10)


		## MODEL options
		self.p.add_argument("--depth_model",
							type=str,
							help="depth model to use",
							choices=["monodepthv2", "litemono"],
							default="litemono")
		self.p.add_argument("--encoder_num_layers",
							type=int,
							help="number of resnet layers",
							default=18,
							choices=[18, 34, 50, 101, 152])
		self.p.add_argument("--weights_init",
							type=str,
							help="pretrained or scratch",
							default="pretrained",
							choices=["pretrained", "scratch"])
		self.p.add_argument("--scales",
							nargs="+",
							type=int,
							help="scales of reconstruction used in the loss",
							default=None)


		# TRAINING options
		self.p.add_argument("--frame_ids",
							nargs="+",
							type=int,
							help="frames to load",
							default=[0, -1, 1])
		self.p.add_argument("--min_depth",
							type=float,
							help="minimum depth",
							default=0.1)
		self.p.add_argument("--max_depth",
							type=float,
							help="maximum depth",
							default=100.0)
		self.p.add_argument("--train_img_type",
							type=str,
							help="type of images to be loadded ",
							choices=["original", "downsample"],
							default=None)


		# Ground Plane Estimation Parameters
		self.p.add_argument("--gp_prior",
							type=float,
							help="ground prior for RANSCA ground estimation",
							default=0.4)
		self.p.add_argument("--gp_tol",
							type=float,
							help="TOL for RANSAC ground estimation",
							default=0.005)
		self.p.add_argument("--gp_max_it",
							type=int,
							help="Maximum iteration for RANSCA ground estimation",
							default=100)
		self.p.add_argument("--gp_np_per_it",
							type=int,
							help="Number of points per iteration for RANSCA ground estimation",
							default=5)	
		

		# LOADING options
		self.p.add_argument("--load_ckpt", "-l",
							type=str,
							help="name of model to load",
							default="")


		# LOGGING options
		self.p.add_argument("--log_frequency",
							type=int,
							help="number of batches between each wandb log",
							default=100)
		self.p.add_argument("--no_train_vis",
							help="if set, train without wandb visualization",
							action="store_true")
		self.p.add_argument("--save_frequency",
							type=int,
							help="number of epochs between each save",
							default=1)
		self.p.add_argument("--comment", "-c",
							type=str,
							help="additional comment wrt experiment",
							default="")
		self.p.add_argument("--print_opt",
							type=bool,
							help="boolean: print the list of opt in command line",
							default=True)
		

		# EVAL options
		self.p.add_argument("--eval_min_depth",
							type=float,
							help="minimum depth used for depth evaluation",
							default=1e-3)
		self.p.add_argument("--eval_max_depth",
							type=float,
							help="maximum depth used for depth evaluation",
							default=None)
		self.p.add_argument("--eval_img_bound",
							nargs="+",
							type=int,
							help="four 0 <= float <= 1 ref. top, bottom, left, right img bound",
							default=None)
		self.p.add_argument("--eval_img_ext",
							type=str,
							help="extension of images to be loadded",
							choices=[".png", ".jpg"],
							default=None)
		self.p.add_argument("--eval_img_type",
							type=str,
							help="type of images to be loadded ",
							choices=["original", "downsample"],
							default=None)
	
	def parse(self, **kwargs):
		self.opt = self.p.parse_args(**kwargs)

		# defines dataset-dependent configs - if they are set via arguments, then the following will be ignored 
		dataset_conf = {
			'split': 			{"waymo": "waymo", "nuscenes": "nuscenes", "kitti": "eigen_zhou"},		# train/test split name
			'height': 			{"waymo": 320, "nuscenes": 288, "kitti": 192},							# height used for training
			'width': 			{"waymo": 480, "nuscenes": 512, "kitti": 640},							# width used for training
			'cam_name': 		{"waymo": 'FRONT', "nuscenes": 'FRONT', "kitti": 'image_02'},			# only used for eval in kitti 
			'train_img_type': 	{"waymo": 'downsample', "nuscenes": 'downsample', "kitti": 'downsample'},#image type used for training
			'eval_max_depth': 	{"waymo": 75, "nuscenes": 75, "kitti": 80},								# max depth bound used for evaluation
			'eval_img_bound': 	{"waymo": 	  [0, 1, 0, 1], 											# image bound used for evaluation 
								 "nuscenes":  [0, 1, 0, 1], 
								 "kitti": 	  [0.40810811, 0.99189189, 0.03594771,  0.96405229]},		# https://github.com/nianticlabs/monodepth2/blob/b676244e5a1ca55564eb5d16ab521a48f823af31/evaluate_depth.py#L193
			'eval_img_ext': 	{"waymo": '.jpg', "nuscenes": '.jpg', "kitti": '.png'},					
			'eval_img_type': 	{"waymo": 'downsample', "nuscenes": 'downsample', "kitti": 'original'},	
		}
		
		if self.opt.scales is None:
			if self.opt.depth_model == "monodepthv2":
				# https://github.com/nianticlabs/monodepth2/blob/b676244e5a1ca55564eb5d16ab521a48f823af31/options.py#L68C23-L68C23
				self.opt.scales = [0, 1, 2, 3]
			if self.opt.depth_model == "litemono":
				# https://github.com/noahzn/Lite-Mono/blob/286c4ab750ad491f0930791b29e401e4882f20d9/options.py#L75C41-L75C41
				self.opt.scales = [0, 1, 2]
		
		if self.opt.data_path is None:
			self.opt.data_path = f"data_dir/{self.opt.dataset}/"

		for k, v in self.opt.__dict__.items():
			if v is None:
				self.opt.__dict__[k] = dataset_conf[k][self.opt.dataset]

		return self.opt
