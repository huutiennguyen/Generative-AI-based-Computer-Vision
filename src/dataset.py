import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import kornia.augmentation as K
import random

def load_pairs(lr_dir: str, hr_dir: str):
    """
    Loads image pairs (LR, HR) from specified directories.

    Args:
        - lr_dir (str): Path to the directory containing low-resolution images.
        - hr_dir (str): Path to the directory containing high-resolution images.

    Returns:
        - list[dict]: A list of dictionaries, where each dictionary contains
                    'lr' and 'hr' image arrays and the 'filename'.
    """
    if not os.path.isdir(lr_dir) or not os.path.isdir(hr_dir):
        raise ValueError(f"One or both directories do not exist: {lr_dir}, {hr_dir}")

    lr_files = {f.replace('.npy', ''): f for f in os.listdir(lr_dir) if f.endswith('.npy')}
    hr_files = {f.replace('.npy', ''): f for f in os.listdir(hr_dir) if f.endswith('.npy')}

    common_ids = sorted(set(hr_files.keys()) & set(lr_files.keys()))
    print(f"Found {len(common_ids)} matched lr-hr pairs.")

    all_pairs = []
    for id_ in common_ids:
        lr_path = os.path.join(lr_dir, lr_files[id_])
        hr_path = os.path.join(hr_dir, hr_files[id_])

        try:
            lr_img = np.load(lr_path)
            hr_img = np.load(hr_path)
            all_pairs.append({'lr': lr_img, 'hr': hr_img, 'filename': hr_files[id_]})
        except Exception as e:
            print(f"Warning: Could not load pair for ID {id_}. Error: {e}")

    return all_pairs

class MRITransform2D:
    def __init__(self, config):
        self.config = config

        # Pre-initialize transforms that require instantiation
        self.elastic = None
        if config.get('elastic', {}).get('enabled', False):
            alpha = config['elastic'].get('alpha', 30.0)
            sigma = config['elastic'].get('sigma', 5.0)
            self.elastic = K.RandomElasticTransform(
                alpha=alpha, sigma=sigma, p=1.0, same_on_batch=True
            )

        self.cutout = None
        if config.get('cutout', {}).get('enabled', False):
            self.cutout = T.RandomErasing(
                p=1.0,
                scale=config['cutout'].get('scale', (0.02, 0.2)),
                ratio=config['cutout'].get('ratio', (0.3, 3.3)),
                value='random'
            )

    def __call__(self, lr_tensor, hr_tensor):
        # Affine transform
        if self.config.get('affine', {}).get('enabled', False):
            affine_cfg = self.config['affine']
            affine = T.RandomAffine(
                degrees=(-affine_cfg.get('degrees', 10), affine_cfg.get('degrees', 10)),
                translate=(affine_cfg.get('translate', 0.02), affine_cfg.get('translate', 0.02)),
                scale=(1 - affine_cfg.get('scale', 0.1), 1 + affine_cfg.get('scale', 0.1)),
                shear=(-affine_cfg.get('shear', 5), affine_cfg.get('shear', 5))
            )
            stacked = torch.stack([lr_tensor, hr_tensor])
            stacked = affine(stacked)
            lr_tensor, hr_tensor = stacked[0], stacked[1]

        # RandomHorizontalFlip
        if self.config.get('hflip', {}).get('enabled', False):
            prob = self.config['hflip'].get('prob', 0.5)
            if random.random() < prob:
                lr_tensor = T.functional.hflip(lr_tensor)
                hr_tensor = T.functional.hflip(hr_tensor)

        # Contrast adjustment
        if self.config.get('contrast', {}).get('enabled', False):
            factor_range = self.config['contrast'].get('factor_range', [0.8, 1.2])
            factor = random.uniform(*factor_range)
            lr_tensor = (lr_tensor - lr_tensor.mean()) * factor + lr_tensor.mean()
            lr_tensor = torch.clamp(lr_tensor, -1.0, 1.0)

        # Brightness adjustment
        if self.config.get('brightness', {}).get('enabled', False):
            delta = self.config['brightness'].get('delta', 0.1)
            shift = random.uniform(-delta, delta)
            lr_tensor = lr_tensor + shift
            lr_tensor = torch.clamp(lr_tensor, -1.0, 1.0)

        # Bias field simulation
        if self.config.get('bias_field', {}).get('enabled', False):
            strength = self.config['bias_field'].get('strength', 0.3)
            bias = self._generate_bias_field(lr_tensor.shape[-2:], strength)
            lr_tensor = lr_tensor * bias
            lr_tensor = torch.clamp(lr_tensor, -1.0, 1.0)

        # Gaussian noise
        if self.config.get('gaus_noise', {}).get('enabled', False):
            std_range = self.config['gaus_noise'].get('std_range', [0.01, 0.05])
            noise_std = random.uniform(*std_range)
            noise = torch.randn_like(lr_tensor) * noise_std
            lr_tensor = lr_tensor + noise
            lr_tensor = torch.clamp(lr_tensor, -1.0, 1.0)

        # Rician noise
        if self.config.get('rician_noise', {}).get('enabled', False):
            std = self.config['rician_noise'].get('std', 0.05)
            # Normalize to [0, 1] to stimulate magnitude
            normed = (lr_tensor + 1.0) / 2.0
            noise_real = torch.randn_like(normed) * std
            noise_imag = torch.randn_like(normed) * std
            rician = torch.sqrt((normed + noise_real)**2 + noise_imag**2)
            # Renormalize to [-1, 1]
            rician = (rician - rician.min()) / (rician.max() - rician.min())
            lr_tensor = rician * 2.0 - 1.0

        # Elastic deformation
        if self.elastic is not None:
            stacked = torch.stack([lr_tensor, hr_tensor])
            stacked = self.elastic(stacked)
            lr_tensor, hr_tensor = stacked[0], stacked[1]

        # Cutout
        if self.cutout is not None:
            lr_tensor = self.cutout(lr_tensor)

        return lr_tensor, hr_tensor

    def _generate_bias_field(self, shape, strength=0.3):
        h, w = shape
        freq = 8
        x = torch.linspace(-1, 1, w)
        y = torch.linspace(-1, 1, h)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        bias = 1.0 + strength * torch.sin(freq * grid_x) * torch.cos(freq * grid_y)
        return bias.to(torch.float32)
    
class MRIDataset2D(Dataset):
    def __init__(self, all_pairs, transform=None):
        self.all_pairs = all_pairs
        self.transform = transform
        self.target_size = (256, 256)

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, idx):
        pair = self.all_pairs[idx]  
        lr_img = pair['lr']
        hr_img = pair['hr']
        filename = pair['filename']

        lr_tensor = torch.from_numpy(lr_img.astype(np.float32)).unsqueeze(0)
        hr_tensor = torch.from_numpy(hr_img.astype(np.float32)).unsqueeze(0)

        if lr_tensor.shape[-2:] != self.target_size:
            lr_tensor = F.interpolate(
                lr_tensor.unsqueeze(0),
                size=self.target_size,
                mode='bicubic',
                align_corners=False,
                antialias=True
            ).squeeze(0)

        if hr_tensor.shape[-2:] != self.target_size:
            hr_tensor = F.interpolate(
                hr_tensor.unsqueeze(0),
                size=self.target_size,
                mode='bicubic',
                align_corners=False,
                antialias=True
            ).squeeze(0)

        # Normalize to [-1, 1]
        lr_tensor = (lr_tensor / 255.0) * 2.0 - 1.0
        hr_tensor = (hr_tensor / 255.0) * 2.0 - 1.0

        # Apply transform if available
        if self.transform:
            lr_tensor, hr_tensor = self.transform(lr_tensor, hr_tensor)

        return {'lr': lr_tensor, 'hr': hr_tensor, 'filename': filename}

def get_dataloader(all_pairs, batch_size, shuffle=True, num_workers=4, transform=None):
    dataset = MRIDataset2D(
        all_pairs=all_pairs,
        transform=transform
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader