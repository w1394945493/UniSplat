# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


def position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100) -> torch.Tensor:
    """
    Convert 2D position grid (HxWx2) to sinusoidal embeddings (HxWxC)

    Args:
        pos_grid: Tensor of shape (H, W, 2) containing 2D coordinates
        embed_dim: Output channel dimension for embeddings

    Returns:
        Tensor of shape (H, W, embed_dim) with positional embeddings
    """
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)  # Flatten to (H*W, 2)

    # Process x and y coordinates separately
    emb_x = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)  # [1, H*W, D/2]
    emb_y = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)  # [1, H*W, D/2]

    # Combine and reshape
    emb = torch.cat([emb_x, emb_y], dim=-1)  # [1, H*W, D]

    return emb.view(H, W, embed_dim)  # [H, W, D]


def make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100) -> torch.Tensor:
    """
    This function generates a 1D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - pos: The position to generate the embedding from.

    Returns:
    - emb: The generated 1D positional embedding.
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.double, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.float()


# Inspired by https://github.com/microsoft/moge


def create_uv_grid(
    width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None
) -> torch.Tensor:
    """
    Create a normalized UV grid of shape (width, height, 2).

    The grid spans horizontally and vertically according to an aspect ratio,
    ensuring the top-left corner is at (-x_span, -y_span) and the bottom-right
    corner is at (x_span, y_span), normalized by the diagonal of the plane.

    Args:
        width (int): Number of points horizontally.
        height (int): Number of points vertically.
        aspect_ratio (float, optional): Width-to-height ratio. Defaults to width/height.
        dtype (torch.dtype, optional): Data type of the resulting tensor.
        device (torch.device, optional): Device on which the tensor is created.

    Returns:
        torch.Tensor: A (width, height, 2) tensor of UV coordinates.
    """
    # Derive aspect ratio if not explicitly provided
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)

    # Compute normalized spans for X and Y
    diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor

    # Establish the linspace boundaries
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height

    # Generate 1D coordinates
    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)

    # Create 2D meshgrid (width x height) and stack into UV
    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    uv_grid = torch.stack((uu, vv), dim=-1)

    return uv_grid


import utils3d
import math
def mask_aware_nearest_resize(
    inputs,
    mask: torch.BoolTensor, 
    size, 
    return_index: bool = False
):
    """
    Resize 2D map by nearest interpolation. Return the nearest neighbor index and mask of the resized map.

    ### Parameters
    - `inputs`: a single or a list of input 2D map(s) of shape (..., H, W, ...). 
    - `mask`: input 2D mask of shape (..., H, W)
    - `size`: target size (target_width, target_height)

    ### Returns
    - `*resized_maps`: resized map(s) of shape (..., target_height, target_width, ...). 
    - `resized_mask`: mask of the resized map of shape (..., target_height, target_width)
    - `nearest_idx`: if return_index is True, nearest neighbor index of the resized map of shape (..., target_height, target_width) for each dimension, .
    """
    height, width = mask.shape[-2:]
    target_width, target_height = size
    device = mask.device
    filter_h_f, filter_w_f = max(1, height / target_height), max(1, width / target_width)
    filter_h_i, filter_w_i = math.ceil(filter_h_f), math.ceil(filter_w_f)
    filter_size = filter_h_i * filter_w_i
    padding_h, padding_w = filter_h_i // 2 + 1, filter_w_i // 2 + 1

    # Window the original mask and uv
    uv = utils3d.torch.image_pixel_center(width=width, height=height, dtype=torch.float32, device=device)
    indices = torch.arange(height * width, dtype=torch.long, device=device).reshape(height, width)
    padded_uv = torch.full((height + 2 * padding_h, width + 2 * padding_w, 2), 0, dtype=torch.float32, device=device)
    padded_uv[padding_h:padding_h + height, padding_w:padding_w + width] = uv
    padded_mask = torch.full((*mask.shape[:-2], height + 2 * padding_h, width + 2 * padding_w), False, dtype=torch.bool, device=device)
    padded_mask[..., padding_h:padding_h + height, padding_w:padding_w + width] = mask
    padded_indices = torch.full((height + 2 * padding_h, width + 2 * padding_w), 0, dtype=torch.long, device=device)
    padded_indices[padding_h:padding_h + height, padding_w:padding_w + width] = indices
    windowed_uv = utils3d.torch.sliding_window_2d(padded_uv, (filter_h_i, filter_w_i), 1, dim=(0, 1))
    windowed_mask = utils3d.torch.sliding_window_2d(padded_mask, (filter_h_i, filter_w_i), 1, dim=(-2, -1))
    windowed_indices = utils3d.torch.sliding_window_2d(padded_indices, (filter_h_i, filter_w_i), 1, dim=(0, 1))

    # Gather the target pixels's local window
    target_uv = utils3d.torch.image_uv(width=target_width, height=target_height, dtype=torch.float32, device=device) * torch.tensor([width, height], dtype=torch.float32, device=device)
    target_lefttop = target_uv - torch.tensor((filter_w_f / 2, filter_h_f / 2), dtype=torch.float32, device=device)
    target_window = torch.round(target_lefttop).long() + torch.tensor((padding_w, padding_h), dtype=torch.long, device=device)

    target_window_uv = windowed_uv[target_window[..., 1], target_window[..., 0], :, :, :].reshape(target_height, target_width, 2, filter_size)                          # (target_height, tgt_width, 2, filter_size)
    target_window_mask = windowed_mask[..., target_window[..., 1], target_window[..., 0], :, :].reshape(*mask.shape[:-2], target_height, target_width, filter_size)     # (..., target_height, tgt_width, filter_size)
    target_window_indices = windowed_indices[target_window[..., 1], target_window[..., 0], :, :].reshape(target_height, target_width, filter_size)                      # (target_height, tgt_width, filter_size)
    target_window_indices = target_window_indices.expand_as(target_window_mask)

    # Compute nearest neighbor in the local window for each pixel 
    dist = torch.where(target_window_mask, torch.norm(target_window_uv - target_uv[..., None], dim=-2), torch.inf)  # (..., target_height, tgt_width, filter_size)
    nearest = torch.argmin(dist, dim=-1, keepdim=True)                                                              # (..., target_height, tgt_width, 1)
    nearest_idx = torch.gather(target_window_indices, index=nearest, dim=-1).squeeze(-1)                            # (..., target_height, tgt_width)
    target_mask = torch.any(target_window_mask, dim=-1)
    nearest_i, nearest_j = nearest_idx // width, nearest_idx % width
    batch_indices = [torch.arange(n, device=device).reshape([1] * i + [n] + [1] * (mask.dim() - i - 1)) for i, n in enumerate(mask.shape[:-2])]
    
    index = (*batch_indices, nearest_i, nearest_j)
    
    if inputs is None:
        outputs = None
    elif isinstance(inputs, torch.Tensor):
        outputs = inputs[index]
    else:
        outputs = tuple(x[index] for x in inputs)
    
    if return_index:
        return outputs, target_mask, index
    else:
        return outputs, target_mask

def scatter_min(size: int, dim: int, index: torch.LongTensor, src: torch.Tensor) -> torch.return_types.min:
    "Scatter the minimum value along the given dimension of `input` into `src` at the indices specified in `index`."
    shape = src.shape[:dim] + (size,) + src.shape[dim + 1:]
    minimum = torch.full(shape, float('inf'), dtype=src.dtype, device=src.device).scatter_reduce(dim=dim, index=index, src=src, reduce='amin', include_self=False)
    minimum_where = torch.where(src == torch.gather(minimum, dim=dim, index=index))
    indices = torch.full(shape, -1, dtype=torch.long, device=src.device)
    indices[(*minimum_where[:dim], index[minimum_where], *minimum_where[dim + 1:])] = minimum_where[dim]
    return torch.return_types.min((minimum, indices))

def split_batch_fwd(fn, chunk_size: int, *args, **kwargs):
    batch_size = next(x for x in (*args, *kwargs.values()) if isinstance(x, torch.Tensor)).shape[0]
    n_chunks = batch_size // chunk_size + (batch_size % chunk_size > 0)
    splited_args = tuple(arg.split(chunk_size, dim=0) if isinstance(arg, torch.Tensor) else [arg] * n_chunks for arg in args)
    splited_kwargs = {k: [v.split(chunk_size, dim=0) if isinstance(v, torch.Tensor) else [v] * n_chunks] for k, v in kwargs.items()}
    results = []
    for i in range(n_chunks):
        chunk_args = tuple(arg[i] for arg in splited_args)
        chunk_kwargs = {k: v[i] for k, v in splited_kwargs.items()}
        results.append(fn(*chunk_args, **chunk_kwargs))

    if isinstance(results[0], tuple):
        return tuple(torch.cat(r, dim=0) for r in zip(*results))
    else:
        return torch.cat(results, dim=0)

def _pad_inf(x_: torch.Tensor):
    return torch.cat([torch.full_like(x_[..., :1], -torch.inf), x_, torch.full_like(x_[..., :1], torch.inf)], dim=-1)

def _pad_cumsum(cumsum: torch.Tensor):
    return torch.cat([torch.zeros_like(cumsum[..., :1]), cumsum, cumsum[..., -1:]], dim=-1)

def _compute_residual(a: torch.Tensor, xyw: torch.Tensor, trunc: float):
    return a.mul(xyw[..., 0]).sub_(xyw[..., 1]).abs_().mul_(xyw[..., 2]).clamp_max_(trunc).sum(dim=-1)

def align(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, trunc = None, eps: float = 1e-7):
    """
    If trunc is None, solve `min sum_i w_i * |a * x_i - y_i|`, otherwise solve `min sum_i min(trunc, w_i * |a * x_i - y_i|)`.
    
    w_i must be >= 0.

    ### Parameters:
    - `x`: tensor of shape (..., n)
    - `y`: tensor of shape (..., n)
    - `w`: tensor of shape (..., n)
    - `trunc`: optional, float or tensor of shape (..., n) or None

    ### Returns:
    - `a`: tensor of shape (...), differentiable
    - `loss`: tensor of shape (...), value of loss function at `a`, detached
    - `index`: tensor of shape (...), where a = y[idx] / x[idx]
    """
    if trunc is None:
        x, y, w = torch.broadcast_tensors(x, y, w)
        sign = torch.sign(x)
        x, y = x * sign, y * sign
        y_div_x = y / x.clamp_min(eps)
        y_div_x, argsort = y_div_x.sort(dim=-1)

        wx = torch.gather(x * w, dim=-1, index=argsort)
        derivatives = 2 * wx.cumsum(dim=-1) - wx.sum(dim=-1, keepdim=True)
        search = torch.searchsorted(derivatives, torch.zeros_like(derivatives[..., :1]), side='left').clamp_max(derivatives.shape[-1] - 1)

        a = y_div_x.gather(dim=-1, index=search).squeeze(-1)
        index = argsort.gather(dim=-1, index=search).squeeze(-1)
        loss = (w * (a[..., None] * x - y).abs()).sum(dim=-1)
        
    else:
        # Reshape to (batch_size, n) for simplicity
        x, y, w = torch.broadcast_tensors(x, y, w)
        batch_shape = x.shape[:-1]
        batch_size = math.prod(batch_shape)
        x, y, w = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1]), w.reshape(-1, w.shape[-1])

        sign = torch.sign(x)
        x, y = x * sign, y * sign
        wx, wy = w * x, w * y
        xyw = torch.stack([x, y, w], dim=-1)    # Stacked for convenient gathering

        y_div_x = A = y / x.clamp_min(eps)
        B = (wy - trunc) / wx.clamp_min(eps)
        C = (wy + trunc) / wx.clamp_min(eps)
        with torch.no_grad():
            # Caculate prefix sum by orders of A, B, C    
            A, A_argsort = A.sort(dim=-1)
            Q_A = torch.cumsum(torch.gather(wx, dim=-1, index=A_argsort), dim=-1)
            A, Q_A = _pad_inf(A), _pad_cumsum(Q_A)    # Pad [-inf, A1, ..., An, inf] and [0, Q1, ..., Qn, Qn] to handle edge cases.

            B, B_argsort = B.sort(dim=-1)
            Q_B = torch.cumsum(torch.gather(wx, dim=-1, index=B_argsort), dim=-1)
            B, Q_B = _pad_inf(B), _pad_cumsum(Q_B)

            C, C_argsort = C.sort(dim=-1)
            Q_C = torch.cumsum(torch.gather(wx, dim=-1, index=C_argsort), dim=-1)
            C, Q_C = _pad_inf(C), _pad_cumsum(Q_C)
            
            # Caculate left and right derivative of A
            j_A = torch.searchsorted(A, y_div_x, side='left').sub_(1)
            j_B = torch.searchsorted(B, y_div_x, side='left').sub_(1)
            j_C = torch.searchsorted(C, y_div_x, side='left').sub_(1)
            left_derivative = 2 * torch.gather(Q_A, dim=-1, index=j_A) - torch.gather(Q_B, dim=-1, index=j_B) - torch.gather(Q_C, dim=-1, index=j_C)
            j_A = torch.searchsorted(A, y_div_x, side='right').sub_(1)
            j_B = torch.searchsorted(B, y_div_x, side='right').sub_(1)
            j_C = torch.searchsorted(C, y_div_x, side='right').sub_(1)
            right_derivative = 2 * torch.gather(Q_A, dim=-1, index=j_A) - torch.gather(Q_B, dim=-1, index=j_B) - torch.gather(Q_C, dim=-1, index=j_C)

            # Find extrema
            is_extrema = (left_derivative < 0) & (right_derivative >= 0)
            is_extrema[..., 0] |= ~is_extrema.any(dim=-1)                       # In case all derivatives are zero, take the first one as extrema.
            where_extrema_batch, where_extrema_index = torch.where(is_extrema)          

            # Calculate objective value at extrema
            extrema_a = y_div_x[where_extrema_batch, where_extrema_index]               # (num_extrema,)
            MAX_ELEMENTS = 4096 ** 2      # Split into small batches to avoid OOM in case there are too many extrema.(~1G)
            SPLIT_SIZE = MAX_ELEMENTS // x.shape[-1]
            extrema_value = torch.cat([
                _compute_residual(extrema_a_split[:, None], xyw[extrema_i_split, :, :], trunc)
                for extrema_a_split, extrema_i_split in zip(extrema_a.split(SPLIT_SIZE), where_extrema_batch.split(SPLIT_SIZE))
            ])          # (num_extrema,)
            
            # Find minima among corresponding extrema
            minima, indices = scatter_min(size=batch_size, dim=0, index=where_extrema_batch, src=extrema_value)        # (batch_size,)
            index = where_extrema_index[indices]

        a = torch.gather(y, dim=-1, index=index[..., None]) / torch.gather(x, dim=-1, index=index[..., None]).clamp_min(eps)
        a = a.reshape(batch_shape)
        loss = minima.reshape(batch_shape)
        index = index.reshape(batch_shape)

    return a, loss, index


def align_points_scale_z_shift(points_src, points_tgt, weight, trunc = None):
    """
    Align `points_src` to `points_tgt` with respect to a shared xyz scale and z shift. 
    It is similar to `align_affine` but scale and shift are applied to different dimensions.

    ### Parameters:
    - `points_src: torch.Tensor` of shape (..., N, 3)
    - `points_tgt: torch.Tensor` of shape (..., N, 3)
    - `weights: torch.Tensor` of shape (..., N)

    ### Returns:
    - `scale: torch.Tensor` of shape (...).
    - `shift: torch.Tensor` of shape (..., 3). x and y shifts are zeros.
    """
    dtype, device = points_src.dtype, points_src.device

    # Flatten batch dimensions for simplicity
    batch_shape, n = points_src.shape[:-2], points_src.shape[-2]
    batch_size = math.prod(batch_shape)
    points_src, points_tgt, weight = points_src.reshape(batch_size, n, 3), points_tgt.reshape(batch_size, n, 3), weight.reshape(batch_size, n)

    # Take anchors
    anchor_where_batch, anchor_where_n = torch.where(weight > 0)
    with torch.no_grad():
        zeros = torch.zeros(anchor_where_batch.shape[0], device=device, dtype=dtype)
        points_src_anchor = torch.stack([zeros, zeros, points_src[anchor_where_batch, anchor_where_n, 2]], dim=-1)      # (anchors, 3)
        points_tgt_anchor = torch.stack([zeros, zeros, points_tgt[anchor_where_batch, anchor_where_n, 2]], dim=-1)      # (anchors, 3)

        points_src_anchored = points_src[anchor_where_batch, :, :] - points_src_anchor[..., None, :]    # (anchors, n, 3)
        points_tgt_anchored = points_tgt[anchor_where_batch, :, :] - points_tgt_anchor[..., None, :]    # (anchors, n, 3)
        weight_anchored = weight[anchor_where_batch, :, None].expand(-1, -1, 3)                         # (anchors, n, 3)

        # Solve optimal scale and shift for each anchor
        MAX_ELEMENTS = 2 ** 20
        scale, loss, index = split_batch_fwd(align, MAX_ELEMENTS // n, points_src_anchored.flatten(-2), points_tgt_anchored.flatten(-2), weight_anchored.flatten(-2), trunc)   # (anchors,)

        loss, index_anchor = scatter_min(size=batch_size, dim=0, index=anchor_where_batch, src=loss)    # (batch_size,)

    # Reproduce by indexing for shorter compute graph
    index_2 = index[index_anchor]                               # (batch_size,) [0, 3n)
    index_1 = anchor_where_n[index_anchor] * 3 + index_2 % 3    # (batch_size,) [0, 3n)

    zeros = torch.zeros((batch_size, n), device=device, dtype=dtype)
    points_tgt_00z, points_src_00z = torch.stack([zeros, zeros, points_tgt[..., 2]], dim=-1), torch.stack([zeros, zeros, points_src[..., 2]], dim=-1)
    tgt_1, src_1 = torch.gather(points_tgt_00z.flatten(-2), dim=1, index=index_1[..., None]).squeeze(-1), torch.gather(points_src_00z.flatten(-2), dim=1, index=index_1[..., None]).squeeze(-1)
    tgt_2, src_2 = torch.gather(points_tgt.flatten(-2), dim=1, index=index_2[..., None]).squeeze(-1), torch.gather(points_src.flatten(-2), dim=1, index=index_2[..., None]).squeeze(-1)

    scale = (tgt_2 - tgt_1) / torch.where(src_2 != src_1, src_2 - src_1, 1.0)
    shift = torch.gather(points_tgt_00z, dim=1, index=(index_1 // 3)[..., None, None].expand(-1, -1, 3)).squeeze(-2) - scale[..., None] * torch.gather(points_src_00z, dim=1, index=(index_1 // 3)[..., None, None].expand(-1, -1, 3)).squeeze(-2)
    scale, shift = scale.reshape(batch_shape), shift.reshape(*batch_shape, 3)

    return scale, shift


def normalize_intrinsics(K, width, height):
    """Normalize intrinsic matrix to have focal length relative to image size."""
    K_norm = K.clone()
    K_norm[0, 0] /= width
    K_norm[1, 1] /= height
    K_norm[0, 2] /= width
    K_norm[1, 2] /= height
    return K_norm


def is_point_in_frustum_batch(points_3d, K, R_c2w, t_c2w, near, far, image_width, image_height):
    N = points_3d.shape[0]
    M = K.shape[0]
    
    R_w2c = R_c2w.transpose(-2, -1)  # (M, 3, 3)
    t_w2c = -torch.bmm(R_w2c, t_c2w.unsqueeze(-1)).squeeze(-1)  # (M, 3)
    
    points_expanded = points_3d.unsqueeze(1).expand(N, M, 3)  # (N, M, 3)
    R_expanded = R_w2c.unsqueeze(0).expand(N, M, 3, 3)  # (N, M, 3, 3)
    t_expanded = t_w2c.unsqueeze(0).expand(N, M, 3)  # (N, M, 3)

    points_cam = torch.matmul(R_expanded, points_expanded.unsqueeze(-1)).squeeze(-1) + t_expanded
    
    z = points_cam[:, :, 2]
    depth_mask = (z >= near) & (z <= far) & (z > 1e-8)
    K_expanded = K.unsqueeze(0).expand(N, M, 3, 3)  # (N, M, 3, 3)
    points_2d_homo = torch.matmul(K_expanded, points_cam.unsqueeze(-1)).squeeze(-1)  # (N, M, 3)
    
    z_homo = points_2d_homo[:, :, 2]
    valid_projection = (z_homo.abs() > 1e-8) & depth_mask
    
    points_2d = torch.zeros_like(points_2d_homo[:, :, :2])
    if valid_projection.any():
        points_2d[valid_projection] = points_2d_homo[valid_projection][:, :2] / z_homo[valid_projection].unsqueeze(-1)
    
    
    image_mask = (points_2d[:, :, 0] >= 0) & (points_2d[:, :, 0] < image_width) & \
                 (points_2d[:, :, 1] >= 0) & (points_2d[:, :, 1] < image_height)
    
    final_mask = valid_projection & image_mask
    
    return final_mask, points_2d.int()