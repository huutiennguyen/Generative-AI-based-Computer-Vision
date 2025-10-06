import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import from our source code modules
from dataset import load_pairs, get_dataloader
from model import CFMUNet
from solver import solve_ode

def test(config: dict, mode: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    system_params = config['system_params'][mode]
    model_params = config['model_params']
    test_params = config['test_params']

    output_pred_dir = system_params['output_pred_path']; os.makedirs(output_pred_dir, exist_ok=True)
    output_lr_dir = system_params['output_lr_path']; os.makedirs(output_lr_dir, exist_ok=True)
    output_hr_dir = system_params['output_hr_path']; os.makedirs(output_hr_dir, exist_ok=True)
    checkpoint_path = system_params['checkpoint_path']; os.makedirs(checkpoint_path, exist_ok=True)

    model = CFMUNet(**model_params).to(device)
    try:
        checkpoint_file = os.path.join(checkpoint_path, "best_model.pth")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded model from: {checkpoint_file}")

    except FileNotFoundError:
        print(f"Error: Could not find 'best_model.pth' in '{checkpoint_path}'.")
        return None, None

    except Exception as e:
        print(f"An error occurred while loading the checkpoint: {e}")
        return None, None
    
    test_set = load_pairs(lr_dir=system_params['lr_dataset_path'], hr_dir=system_params['hr_dataset_path'])
    test_loader = get_dataloader(test_set, batch_size=test_params['batch_size'], shuffle=False, transform=None)
    model.eval()
    all_images = []

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Inference", colour="#219ebc")
        for batch in progress_bar:
            lr_imgs_norm = batch['lr'].to(device)
            hr_imgs_norm = batch['hr'].to(device)
            filenames = batch['filename']
            pred_imgs_norm = solve_ode(model, cond_img=lr_imgs_norm, t_steps=test_params['t_steps'],device=device)

            for i in range(pred_imgs_norm.shape[0]):
                lr_img_norm = lr_imgs_norm[i]
                hr_img_norm = hr_imgs_norm[i]
                pred_img_norm = pred_imgs_norm[i]
                current_filename = filenames[i]

                lr_img_to_save = ((lr_img_norm + 1) * 0.5 * 255.0).cpu().numpy()
                hr_img_to_save = ((hr_img_norm + 1) * 0.5 * 255.0).cpu().numpy()
                pred_img_to_save = ((pred_img_norm + 1) * 0.5 * 255.0).cpu().numpy()
                
                # Get base filename (without .mat extension)
                base_filename = os.path.splitext(current_filename)[0]

                # Save images as .png files
                plt.imsave(os.path.join(output_lr_dir, f"{base_filename}.png"), lr_img_to_save.squeeze(), cmap='gray')
                plt.imsave(os.path.join(output_hr_dir, f"{base_filename}.png"), hr_img_to_save.squeeze(), cmap='gray')
                plt.imsave(os.path.join(output_pred_dir, f"{base_filename}.png"), pred_img_to_save.squeeze(), cmap='gray')

                # Save images as .npy files
                np.save(os.path.join(output_lr_dir, f"{base_filename}.npy"), lr_img_to_save.squeeze())
                np.save(os.path.join(output_hr_dir, f"{base_filename}.npy"), hr_img_to_save.squeeze())
                np.save(os.path.join(output_pred_dir, f"{base_filename}.npy"), pred_img_to_save.squeeze())

                image_dict = {
                        'lr': lr_img_to_save,
                        'hr': hr_img_to_save,
                        'pred': pred_img_to_save
                }
                all_images.append(image_dict)
    print(f"\n--- Inference process completed! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Conditional Flow Matching model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    parser.add_argument('--mode', type=str, required=True, choices=['ind_testing_phase', 'ood_testing_phase'],
                        help="The evaluation mode to run ('ind_testing_phase' or 'ood_testing_phase').")
    parser.add_argument('--batch_size', type=int, help='Override the batch size in the config file.')
    parser.add_argument('--t_steps', type=int, help='Override the t_steps in the config file.')
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        exit()
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        exit()

    if args.batch_size is not None:
        config['test_params']['batch_size'] = args.batch_size
        print(f"Overriding batch_size with command-line value: {args.batch_size}")
    if args.t_steps is not None:
        config['test_params']['t_steps'] = args.t_steps
        print(f"Overriding t_steps with command-line value: {args.t_steps}")

    test(config=config, mode=args.mode)