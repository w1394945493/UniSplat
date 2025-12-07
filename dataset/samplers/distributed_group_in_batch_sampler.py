import copy
import itertools

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def sync_random_seed(seed=None, device='cuda'):
    """Make sure different ranks share the same seed.

    All workers must call this function, otherwise it will deadlock.
    This method is generally used in `DistributedSampler`,
    because the seed should be identical across all processes
    in the distributed group.
    In distributed sampling, different ranks should sample non-overlapped
    data in the dataset. Therefore, this function is used to make sure that
    each rank shuffles the data indices in the same order based
    on the same seed. Then different ranks could use different indices
    to select non-overlapped data from the same data list.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is None:
        seed = np.random.randint(2**31)
    assert isinstance(seed, int)

    rank, world_size = get_dist_info()

    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


class SamplerState(object):

    def __init__(self, samples_per_gpu: int, epoch_seed: bool = False):
        """
        Args:
            samples_per_gpu (int): batch_size in one gpu
            epoch_seed (bool): if sampler random seed is associated with epoch
        """
        # sampler iter number => samples_per_gpu * trained_it
        self.inner_iter = 0
        self.samples_per_gpu = samples_per_gpu
        self.epoch_seed = epoch_seed
        self._resumed = False

    def __iter__(self):
        raise NotImplementedError

    def set_epoch(self):
        raise NotImplementedError

    def init_generators(self):
        raise NotImplementedError

    @property
    def resumed(self):
        return self._resumed

    def _set_resume_sign(self, sign: bool):
        self._resumed = sign

    def resume_state(self, trained_it: int, trained_epoch: int, epoch_iter_num: int):
        iter_within_epoch = trained_it - (trained_epoch * epoch_iter_num)
        assert iter_within_epoch >= 0, 'iter_within_epoch is supported to greater than or equal to zero'
        if trained_it:
            if self.epoch_seed:
                target_inner_iter = self.samples_per_gpu * iter_within_epoch
            else:
                target_inner_iter = self.samples_per_gpu * trained_it
            for idx, _ in enumerate(self):
                if idx % 10000 == 0:
                    print(f'resume state idx: {idx}', f'inner_iter: {self.inner_iter}', flush=True)
                self.inner_iter += 1
                if self.inner_iter == target_inner_iter:
                    break
            self._resumed = True


class DistributedGroupInBatchSampler(DistributedSampler, SamplerState):
    """Pardon this horrendous name. Basically, we want every sample to be from
    its own group. If batch size is 4 and # of GPUs is 8, each sample of these
    32 should be operating on its own group.

    Shuffling is only done for group order, not done within groups.
    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        world_size=None,
        rank=None,
        seed=0,
        skip_prob=0.0,
        sequence_flip_prob=0.0,
        shuffle=True,
        num_replicas=None,
        drop_last=False,
    ):
        _rank, _world_size = get_dist_info()
        if world_size is None:
            world_size = _world_size
        if rank is None:
            rank = _rank
        if num_replicas is None:
            num_replicas = world_size
        super().__init__(
            dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
        epoch_seed = hasattr(self, '_set_random_seed')
        SamplerState.__init__(self, samples_per_gpu=batch_size, epoch_seed=epoch_seed)
        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.seed = sync_random_seed(seed)
        self.shuffle = shuffle

        assert hasattr(self.dataset, 'seq_flag')
        self.seq_flag = self.dataset.seq_flag
        self.group_sizes = np.bincount(self.seq_flag)
        self.groups_num = len(self.group_sizes)
        self.global_batch_size = batch_size * world_size
        assert self.groups_num >= self.global_batch_size, f'The number of sequences ({self.groups_num}) is less than the global batch size ({self.global_batch_size})'

        # Now, for efficiency, make a dict group_idx: List[dataset sample_idxs]
        if hasattr(self.dataset, 'group_idx_to_sample_idxs') and self.dataset.group_idx_to_sample_idxs:
            self.group_idx_to_sample_idxs = self.dataset.group_idx_to_sample_idxs
        else:
            self.group_idx_to_sample_idxs = {
                group_idx: np.where(self.seq_flag == group_idx)[0].tolist()
                for group_idx in range(self.groups_num)
            }

        # Only for evaluation
        self.samples_per_batch = self._compute_samples_per_batch()
        self.max_iters_per_batch = np.max(self.samples_per_batch)
        self._compute_group_permutation()
        self.init_generators()
        self.skip_prob = skip_prob
        self.sequence_flip_prob = sequence_flip_prob

    def _compute_group_permutation(self):
        """Builds self._perm as a list of group indices once per epoch."""
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed)
            self._perm = torch.randperm(self.groups_num, generator=g).tolist()
        else:
            self._perm = list(range(self.groups_num))

    def init_generators(self):
        # Get a generator per sample idx. Considering samples over all
        # GPUs, each sample position has its own generator
        self.group_indices_per_global_sample_idx = [
            self._group_indices_per_global_sample_idx(self.rank * self.batch_size + local_sample_idx)
            for local_sample_idx in range(self.batch_size)
        ]

        # Keep track of a buffer of dataset sample idxs for each local sample idx
        self.buffer_per_local_sample = [[] for _ in range(self.batch_size)]

    def _compute_samples_per_batch(self):
        """Compute the number of samples per batch."""
        samples_per_group = [len(sample_idxs) for sample_idxs in self.group_idx_to_sample_idxs.values()]
        total_batch_size = self.batch_size * self.world_size
        samples_per_batch = [0] * total_batch_size

        for i, sample_count in enumerate(samples_per_group):
            samples_per_batch[i % total_batch_size] += sample_count

        return samples_per_batch

    def _infinite_group_indices(self):
        while True:
            yield from self._perm

    def _group_indices_per_global_sample_idx(self, global_sample_idx):
        yield from itertools.islice(
            self._infinite_group_indices(),
            global_sample_idx,
            None,
            self.global_batch_size,
        )

    def __iter__(self):
        while True:
            for local_sample_idx in range(self.batch_size):
                skip = (
                    np.random.uniform() < self.skip_prob and len(self.buffer_per_local_sample[local_sample_idx]) > 1)
                if len(self.buffer_per_local_sample[local_sample_idx]) == 0:
                    # Finished current group, refill with next group
                    # skip = False
                    new_group_idx = next(self.group_indices_per_global_sample_idx[local_sample_idx])
                    self.buffer_per_local_sample[local_sample_idx] = copy.deepcopy(
                        self.group_idx_to_sample_idxs[new_group_idx])
                    if np.random.uniform() < self.sequence_flip_prob:
                        self.buffer_per_local_sample[local_sample_idx] = self.buffer_per_local_sample[
                            local_sample_idx][::-1]

                if skip:
                    self.buffer_per_local_sample[local_sample_idx].pop(0)
                idx = self.buffer_per_local_sample[local_sample_idx].pop(0)

                yield idx

    def __len__(self):
        """Length of base dataset."""
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._set_random_seed(self.epoch)
        self._compute_group_permutation()
        if not self.resumed:
            self.init_generators()
        else:
            self._set_resume_sign(False)

    def _set_random_seed(self, seed):
        self.seed = seed

