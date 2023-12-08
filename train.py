import os, torch
from torch.distributed import init_process_group, destroy_process_group
from options import DynamoOptions
from Trainer import Trainer

def ddp_setup():
    init_process_group(backend="nccl")

def ddp_cleanup():
    destroy_process_group()
    
if __name__ == "__main__":

    options = DynamoOptions()
    opt = options.parse()

    ## CHECK FOR DISTRIBUTED TRAINING

    opt.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))    # set to 1 if not found
    opt.ddp = opt.local_world_size > 1
    assert len(opt.cuda_ids) == opt.local_world_size, f'opt.cuda_ids(={opt.cuda_ids}) does not match opt.local_world_size(={opt.local_world_size})'


    ## START TRAINING

    if opt.ddp:
        ddp_setup()     ## setup distributed training
    
    trainer = Trainer(opt)
    trainer.train()

    if opt.ddp:
        ddp_cleanup()   ## clean up distributed training
