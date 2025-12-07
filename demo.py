import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import argparse
from omegaconf import OmegaConf
import numpy as np
from accelerate.utils import set_seed
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.amp import autocast
from safetensors.torch import load_file

from dataset.waymo import WaymoDataset
from dataset.utils import Logger
from pi3.models.pi3 import Pi3
from dataset.samplers.distributed_group_in_batch_sampler import DistributedGroupInBatchSampler, get_dist_info
import model.gaussian_head as gaussian_head_class

def parse_args():
    parser = argparse.ArgumentParser(description="Training script with YAML config")
    parser.add_argument("--config", type=str, default='configs/waymo.yaml', help="Path to the YAML config file")
    parser.add_argument("--work_dir_root", type=str, default="./work_dirs", help="Root directory for storing outputs")
    parser.add_argument("--load_from", type=str, default=None, help="ckpt load from")
    parser.add_argument("--data_path", type=str, default='data/waymo', help="data load from")
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', 
                        help='job launcher: none for no distributed, pytorch for torchrun, accelerate for Accelerator')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training')
    return parser.parse_args()

def init_distributed_mode(args):
    if args.launcher == 'none':
        return
    elif args.launcher == 'pytorch':
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.local_rank = int(os.environ['LOCAL_RANK'])
        else:
            print('Not using distributed mode')
            args.distributed = False
            return
        args.distributed = True
        torch.cuda.set_device(args.local_rank)
        args.dist_backend = 'nccl'
        print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        dist.barrier()


def save_img(tmp, save_path):
    tmp = tmp.permute(1,2,0)
    tmp = tmp.cpu().numpy()
    tmp = tmp * 255
    tmp = tmp.astype(np.uint8)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)
    tmp = cv2.imwrite(save_path, tmp)


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    work_dir = os.path.join(args.work_dir_root, 'demo')
    os.makedirs(work_dir, exist_ok=True)
    
    # Setup distributed training
    args.distributed = False
    args.dist_url = 'env://'  # Default for PyTorch distributed
    args.rank = 0
    args.world_size = 1
    init_distributed_mode(args)
    if args.launcher == 'pytorch':
        is_main_process = args.rank == 0
    else:
        is_main_process = True  # Single GPU mode

    set_seed(42)
    logging = Logger(is_main_process, work_dir=work_dir)
    logging.info(cfg)
    # Create dataset and dataloader
    dataset = WaymoDataset(scene_root=args.data_path, is_train=False, cfg=cfg.Dataset)
    rank, world_size = get_dist_info()
    sampler = DistributedGroupInBatchSampler(dataset, batch_size=1, seed=None, rank=rank, world_size=world_size, shuffle=True)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, sampler=sampler,
        num_workers=0, pin_memory=True)
    # Create model
    model = Pi3()
    model_name = cfg.Model.Gaussian_head.Name
    model_class = getattr(gaussian_head_class, model_name)
    model.gaussian_head = model_class(dim_in=2048, cfg=cfg.Model.Gaussian_head)
    model.gaussian_head.image_backbone.mask_token = None
    weight = load_file(args.load_from, device='cpu')
    model.load_state_dict(weight, strict=False)
    device = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Wrap model with DDP if using PyTorch distributed
    if args.distributed and args.launcher == 'pytorch':
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    # start evaluation
    logging.info(f"Dataset loaded, {len(dataset)} samples, {len(dataloader)} iterations")
    amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model.eval()
    with torch.no_grad():
        dataloader_iter = iter(dataloader)
        for batch_idx in range(0, len(dataloader)):
            batch = next(dataloader_iter)            
            for key in batch.keys():
                if key not in ['input_dict_gs', 'output_dict_gs']:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
            images = batch['images'].to(device)

            unwrapped_model = model.module if args.distributed else model
            with autocast(device_type='cuda', dtype=amp_dtype):
                res = model(images)

            with autocast(device_type='cuda', enabled=False):          
                if hasattr(unwrapped_model, 'gaussian_head'):
                    input_dict_gs = batch['input_dict_gs']
                    input_dict_gs['sky_mask'] = batch['sky_mask'].to(images.dtype).to(device)
                    input_dict_gs['single_depthmaps'] = batch['single_depthmaps'].to(device)
                    batch['input_dict_gs']['intrinsics'] = batch['intrinsics'].to(device)
                    batch['input_dict_gs']['camera2lidar'] = batch['camera2lidar'].to(device)
                    batch['input_dict_gs']['dynamics_region'] = batch['dynamics_region'].to(device)
                    res['scene'] = batch['scene']
                    res['frame'] = batch['frame']
                    lidar2world = batch['camera_pose'] @ batch['camera2lidar'].inverse()
                    res['lidar2world'] = lidar2world
                    render_pkg_pixel, _ = unwrapped_model.gaussian_head(res, images, 5, input_dict_gs=batch['input_dict_gs'], output_dict_gs=batch['output_dict_gs'])
                    # save results
                    rgb_pred = render_pkg_pixel['image']
                    rgb_gt = batch['output_dict_gs']["rgb"].to(rgb_pred.device, rgb_pred.dtype)[0]
                    scene = batch['scene'][0]
                    frame = batch['frame'][0]
                    save_path = os.path.join(f'{work_dir}/vis/', f'{scene}_{frame}')
                    os.makedirs(save_path, exist_ok=True)
       
                    dyn = render_pkg_pixel['dyn']
                    for i in range(5, 10):
                        tmp = dyn[i][0].detach().clone().cpu().numpy()
                        tmp = tmp > 0.5
                        tmp = tmp * 255
                        tmp = tmp.astype(np.uint8)
                        tmp = cv2.imwrite(os.path.join(save_path, f'recon_dyn_{i-5}.png'), tmp)
                    for i in range(5):
                        tmp = render_pkg_pixel["image"][i].detach().clone()
                        save_img(tmp, os.path.join(save_path, f'novel_{i}.png'))
                        tmp = rgb_gt[i].detach().clone()
                        save_img(tmp, os.path.join(save_path, f'novel_{i}_gt.png'))
                    for i in range(5, 10):
                        tmp = render_pkg_pixel["image"][i].detach().clone()
                        save_img(tmp, os.path.join(save_path, f'recon_{i-5}.png'))
                        tmp = rgb_gt[i].detach().clone()
                        save_img(tmp, os.path.join(save_path, f'recon_{i-5}_gt.png'))
                   
if __name__ == "__main__":
    main()
    
