import torch
import numpy as np
import os

def get_parameter_groups(model, weight_decay, skip_list=()):
    parameter_groups = [
        {"params": [], "weight_decay": 0.0},
        {"params": [], "weight_decay": weight_decay}
    ]
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue 

        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            parameter_groups[0]["params"].append(param)
        else:
            parameter_groups[1]["params"].append(param)
    
    parameter_groups = [g for g in parameter_groups if len(g["params"]) > 0]
    
    return parameter_groups


def inv(mat):
    """Invert a torch or numpy matrix"""
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f"bad matrix type = {type(mat)}")


def geotrf(Trf, pts, ncol=None, norm=False):
    """Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    if (
        isinstance(Trf, torch.Tensor)
        and isinstance(pts, torch.Tensor)
        and Trf.ndim == 3
        and pts.ndim == 4
    ):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = (
                torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts)
                + Trf[:, None, None, :d, d]
            )
        else:
            raise ValueError(f"bad shape, not ending with 3 or 4, for {pts.shape=}")
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], "batch size does not match"
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:

                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:

                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res


def get_parameter_groups_v2(model, weight_decay, lr, agg_lr):
    parameter_groups = [
        {"params": [], "weight_decay": 0.0, "lr": lr},
        {"params": [], "weight_decay": weight_decay, "lr": lr},
        {"params": [], "weight_decay": 0.0, "lr": agg_lr},
        {"params": [], "weight_decay": weight_decay, "lr": agg_lr}
    ]
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue 
        if 'aggregator' not in name:
            if len(param.shape) == 1 or name.endswith(".bias"):
                parameter_groups[0]["params"].append(param)
            else:
                parameter_groups[1]["params"].append(param)
        else:
            if len(param.shape) == 1 or name.endswith(".bias"):
                parameter_groups[2]["params"].append(param)
            else:
                parameter_groups[3]["params"].append(param)
        
    parameter_groups = [g for g in parameter_groups if len(g["params"]) > 0]
    
    return parameter_groups


def get_parameter_groups_v3(model, weight_decay, key_configs):
    """
    Create parameter groups with different learning rates for different parameter keys.
    
    Args:
        model: The model whose parameters need to be grouped
        weight_decay: Weight decay value for regularization
        key_configs: Dictionary mapping parameter name keys to learning rates
                    Format: {key: lr, key2: lr2, ...}
                    where key is a substring to match in parameter names
    
    Returns:
        List of parameter groups for optimizer
    """
    parameter_groups = []
    
    # Create two groups for each key: one without weight decay (bias/1D), one with weight decay
    for key, lr in key_configs.items():
        parameter_groups.extend([
            {"params": [], "weight_decay": 0.0, "lr": lr, "key": key, "type": "bias"},
            {"params": [], "weight_decay": weight_decay, "lr": lr, "key": key, "type": "weight"}
        ])
    
    # Add default groups for parameters that don't match any key
    default_lr = key_configs.get('default', 1e-3)  # Set default learning rate
    parameter_groups.extend([
        {"params": [], "weight_decay": 0.0, "lr": default_lr, "key": "default", "type": "bias"},
        {"params": [], "weight_decay": weight_decay, "lr": default_lr, "key": "default", "type": "weight"}
    ])
    
    # Assign parameters to corresponding groups
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        matched_key = None
        for key in key_configs.keys():
            if key != 'default' and key in name:
                matched_key = key
                break
        
        if matched_key is None:
            matched_key = 'default'
        
        # Choose the appropriate group based on parameter type
        if len(param.shape) == 1 or name.endswith(".bias"):
            # Find the bias and norm group for the matched key
            for group in parameter_groups:
                if group["key"] == matched_key and group["type"] == "bias":
                    group["params"].append(param)
                    break
        else:
            # Find the weight group for the matched key
            for group in parameter_groups:
                if group["key"] == matched_key and group["type"] == "weight":
                    group["params"].append(param)
                    break
    
    # Filter out empty parameter groups
    parameter_groups = [g for g in parameter_groups if len(g["params"]) > 0]
    
    for group in parameter_groups:
        group.pop("key", None)
        group.pop("type", None)
    
    return parameter_groups


import logging
from datetime import datetime
class Logger:
    def __init__(self, is_main_process, work_dir=None):
        self.is_main_process = is_main_process
        
        if is_main_process and work_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{work_dir}/train_log_{timestamp}.txt"
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler() 
                ]
            )
        
        self.logger = logging.getLogger(__name__)
    
    def info(self, message):
        if self.is_main_process:
            self.logger.info(message)
    
    def warning(self, message):
        if self.is_main_process:
            self.logger.warning(message)
    
    def error(self, message):
        if self.is_main_process:
            self.logger.error(message)
    
    def debug(self, message):
        if self.is_main_process:
            self.logger.debug(message)
    
    def critical(self, message):
        if self.is_main_process:
            self.logger.critical(message)


def check_model_parameters(model, model_path, logging):
    """Check model parameter loading status and whether they were actually loaded"""
    from safetensors.torch import load_file
    import os
    import torch
    import random
    
    # Load original weight files
    if os.path.exists(os.path.join(model_path, "model.safetensors")):
        saved_state_dict = load_file(os.path.join(model_path, "model.safetensors"))
    else:
        # If using sharded files
        saved_state_dict = {}
        for file in os.listdir(model_path):
            if file.endswith('.safetensors'):
                file_dict = load_file(os.path.join(model_path, file))
                saved_state_dict.update(file_dict)
    
    # Get current model parameters
    current_state_dict = model.state_dict()
    
    # Compare keys
    model_keys = set(current_state_dict.keys())
    saved_keys = set(saved_state_dict.keys())
    
    missing_keys = model_keys - saved_keys
    unexpected_keys = saved_keys - model_keys
    matched_keys = model_keys & saved_keys
    
    # Check if matched parameters were actually loaded correctly
    not_loaded_correctly = []
    if matched_keys:
        # Convert to list and randomly sample
        matched_keys_list = list(matched_keys)
        sample_size = min(50, len(matched_keys_list))
        sampled_keys = random.sample(matched_keys_list, sample_size)
        
        for key in sampled_keys:
            if not torch.equal(current_state_dict[key], saved_state_dict[key]):
                not_loaded_correctly.append(key)
    
    logging.info("=== Parameter Loading Check ===")
    logging.info(f"Total model parameters: {len(model_keys)}")
    logging.info(f"Total saved parameters: {len(saved_keys)}")
    logging.info(f"Matched parameters: {len(matched_keys)}")
    logging.info(f"Missing parameters: {len(missing_keys)}")
    logging.info(f"Unexpected parameters: {len(unexpected_keys)}")
    logging.info(f"Parameters not loaded correctly: {len(not_loaded_correctly)}")
    
    if missing_keys:
        logging.warning("Missing parameters:")
        for key in list(missing_keys)[:10]:
            logging.warning(f"  - {key}")
        if len(missing_keys) > 10:
            logging.warning(f"  ... and {len(missing_keys) - 10} more")
    
    if unexpected_keys:
        logging.info("Unexpected parameters:")
        for key in list(unexpected_keys)[:10]:
            logging.info(f"  - {key}")
        if len(unexpected_keys) > 10:
            logging.info(f"  ... and {len(unexpected_keys) - 10} more")
    
    if not_loaded_correctly:
        logging.info("Parameters not loaded correctly:")
        for key in list(not_loaded_correctly)[:10]:
            logging.info(f"  - {key}")
        if len(not_loaded_correctly) > 10:
            logging.info(f"  ... and {len(not_loaded_correctly) - 10} more")
    
    return 

def find_latest_checkpoint(work_dir):
    """
    Find the latest epoch checkpoint in work_dir
    Returns the path of the latest checkpoint, or None if not found
    """
    if not os.path.exists(work_dir):
        return None, None
    
    # Find all model_epoch_* directories
    epoch_dirs = []
    for item in os.listdir(work_dir):
        item_path = os.path.join(work_dir, item)
        if os.path.isdir(item_path) and item.startswith('model_epoch_'):
            try:
                # Extract epoch number
                epoch_num = int(item.split('model_epoch_')[-1])
                epoch_dirs.append((epoch_num, item_path))
            except ValueError:
                continue
    
    if not epoch_dirs:
        return None, None
    
    # Sort by epoch number and return the latest one
    latest_epoch, latest_path = max(epoch_dirs, key=lambda x: x[0])
    
    # Verify that checkpoint directory contains necessary files
    required_files = ['config.json', 'model.safetensors']  # Adjust based on your model save format
    if all(os.path.exists(os.path.join(latest_path, f)) for f in required_files):
        return latest_path, latest_epoch
    else:
        return None, None


import PIL.Image

class ImageList:
    """Convenience class to aply the same operation to a whole set of images."""

    def __init__(self, images):
        if not isinstance(images, (tuple, list, set)):
            images = [images]
        self.images = []
        for image in images:
            if not isinstance(image, PIL.Image.Image):
                image = PIL.Image.fromarray(image)
            self.images.append(image)

    def __len__(self):
        return len(self.images)

    def to_pil(self):
        return tuple(self.images) if len(self.images) > 1 else self.images[0]

    @property
    def size(self):
        sizes = [im.size for im in self.images]
        assert all(sizes[0] == s for s in sizes)
        return sizes[0]

    def resize(self, *args, **kwargs):
        return ImageList(self._dispatch("resize", *args, **kwargs))

    def crop(self, *args, **kwargs):
        return ImageList(self._dispatch("crop", *args, **kwargs))

    def _dispatch(self, func, *args, **kwargs):
        return [getattr(im, func)(*args, **kwargs) for im in self.images]


try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
    nearest = PIL.Image.Resampling.NEAREST
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC

def rescale_image_depthmap_custom(
    image, depthmap, camera_intrinsics, output_resolution, force=True
):
    """Jointly rescale a (image, depthmap)
    so that (out_width, out_height) >= output_res
    """
    image = ImageList(image)
    input_resolution = np.array(image.size)  # (W,H)
    output_resolution = np.array(output_resolution)
    if depthmap is not None:
        raise NotImplementedError

    # define output resolution
    assert output_resolution.shape == (2,)
    scale_w = output_resolution[0] / input_resolution[0]
    scale_h = output_resolution[1] / input_resolution[1]

    # first rescale the image so that it contains the crop
    resample_algo = lanczos if (scale_w < 1 or scale_h < 1) else bicubic
    image = image.resize(output_resolution, resample=resample_algo)
    if depthmap is not None:
        raise NotImplementedError

    # no offset here; simple rescaling
    camera_intrinsics = camera_matrix_of_crop_custom(
        camera_intrinsics, scale_w, scale_h
    )

    return image.to_pil(), depthmap, camera_intrinsics

def depthmap_to_absolute_camera_coordinates(
    depthmap, camera_intrinsics, camera_pose
):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
        - camera_pose: a 4x3 or 4x4 cam2world matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    X_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, camera_intrinsics)

    X_world = X_cam  # default
    if camera_pose is not None:

        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3, 3]

        X_world = (
            np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]
        )

    return X_world, valid_mask

def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_cam = depthmap
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    valid_mask = depthmap > 0.0
    return X_cam, valid_mask


def colmap_to_opencv_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K


def opencv_to_colmap_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K


def camera_matrix_of_crop_custom(
    input_camera_matrix,
    scale_x,
    scale_y,
):
    # Generate new camera parameters
    output_camera_matrix_colmap = opencv_to_colmap_intrinsics(input_camera_matrix)
    output_camera_matrix_colmap[0, 0] *= scale_x  # fx
    output_camera_matrix_colmap[1, 1] *= scale_y  # fy
    output_camera_matrix_colmap[0, 2] *= scale_x  # cx
    output_camera_matrix_colmap[1, 2] *= scale_y 
    output_camera_matrix = colmap_to_opencv_intrinsics(output_camera_matrix_colmap)

    return output_camera_matrix