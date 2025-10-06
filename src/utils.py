import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from IPython.display import clear_output
import lpips
loss_fn = lpips.LPIPS(net='alex').to('cuda' if torch.cuda.is_available() else 'cpu')

def display_images(image_dicts, titles=None, num_rows=1):
    """
    Displays a grid of images from a list of dictionaries (e.g., {'lr':..., 'hr':..., 'pred':...}).

    Args:
        - image_dicts (list of dict): Each element is a dict with image arrays.
        - titles (list of str): Optional list of keys to display from each dict, in order. If None, use all keys sorted alphabetically.
        - num_rows (int): Number of image sets (rows) to display.
    """
    if titles is None:
        # Auto-detect keys from the first dict and sort alphabetically for consistency
        titles = sorted(image_dicts[0].keys())

    num_cols = len(titles)
    sample_indices = random.sample(range(len(image_dicts)), num_rows)

    plt.style.use('ggplot')
    plt.rcParams['font.family'] = 'serif'
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))

    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for row_idx, idx in enumerate(sample_indices):
        data_dict = image_dicts[idx]

        for col_idx, key in enumerate(titles):
            img_data = data_dict[key]

            if isinstance(img_data, torch.Tensor):
                img_display = img_data.squeeze().cpu().numpy()
            else:
                img_display = np.squeeze(img_data)

            ax = axes[row_idx, col_idx]
            ax.imshow(img_display, cmap='gray')
            ax.axis('off')
            ax.set_title(f"{key.upper()} #{idx}, shape: {img_data.shape}", fontsize=12)

    plt.tight_layout()
    plt.show()

def generate_brain_mask(image: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """
    Tạo mask vùng não từ ảnh grayscale bằng threshold + morphological processing.
    """
    img = image.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    binary = (img > threshold).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(image, dtype=np.uint8)

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest_label).astype(np.uint8)

    return mask

def compute_metrics(gt, pred, mask=None, use_mask=False, fixed_range=255.0):
    if use_mask and mask is not None:
        gt = gt * mask
        pred = pred * mask

    ssim_score = ssim(gt, pred, data_range=fixed_range)
    psnr_score = psnr(gt, pred, data_range=fixed_range)
    rmse_score = np.sqrt(mse(gt, pred))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gt_tensor = torch.tensor(gt).unsqueeze(0).unsqueeze(0).float().to(device)
    pred_tensor = torch.tensor(pred).unsqueeze(0).unsqueeze(0).float().to(device)
    lpips_score = loss_fn(gt_tensor, pred_tensor).item()

    return {
        "SSIM": ssim_score,
        "PSNR": psnr_score,
        "RMSE": rmse_score,
        "LPIPS": lpips_score
    }

def plot_live_losses(train_loss_log, val_loss_log):
    """
    Plots the training and validation loss live.
    
    Args:
        - train_loss_log (list): A list of training loss values.
        - val_loss_log (list): A list of validation loss values.
    """
    clear_output(wait=True)
    plt.style.use('ggplot')
    plt.rcParams['font.family'] = 'serif'
    _, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_loss_log, label='Training Loss', color='#219ebc')
    ax.plot(val_loss_log, label='Validation Loss', color='#fb8500')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss Over Time')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()