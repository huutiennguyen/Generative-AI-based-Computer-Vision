import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from einops import rearrange

# Import from our source code modules
from dataset import load_pairs, MRITransform2D, get_dataloader
from model import CFMUNet
from utils import plot_live_losses

def train(params: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    system_params = params['system_params']['training_phase']
    model_params = params['model_params']
    train_params = params['train_params']
    transform_params = params['transform_params']

    checkpoint_path = system_params['checkpoint_path']
    os.makedirs(checkpoint_path, exist_ok=True)

    # init model, data loader, optimizer, scheduler
    model = CFMUNet(**model_params).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    train_transform = MRITransform2D(config=transform_params)

    train_dataset = load_pairs(lr_dir=system_params['lr_dataset_path'], hr_dir=system_params['hr_dataset_path'])
    train_set, val_set = train_test_split(train_dataset, test_size=0.2, random_state=42)
    train_loader = get_dataloader(
        all_pairs=train_set,
        batch_size=train_params['batch_size'],
        shuffle=True,
        num_workers=4,
        transform=train_transform
    )
    val_loader = get_dataloader(
        all_pairs=val_set,
        batch_size=train_params['batch_size'],
        shuffle=False,
        num_workers=4,
        transform=None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_params['learning_rate'], weight_decay=train_params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_params['epochs'] , eta_min=1e-6)

    best_val_loss = float('inf')
    train_loss_log, val_loss_log = [], []
    print(f"Initialized model with {num_params:,} params on {device}. Starting training...\n", "=" * 50, sep='')
    for epoch in tqdm(range(1, train_params['epochs'] + 1)):
        # training
        model.train()
        total_training_loss = 0.0

        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}|Training", leave=True, colour="#219ebc")
        for batch in train_progress_bar:
            optimizer.zero_grad()
            x_lr  = batch['lr'].to(device)
            x_hr = batch['hr'].to(device)
            x_noise = (torch.rand_like(x_hr) * 2) - 1 # normalized to [-1, 1]

            t = torch.rand(x_hr.shape[0], device=x_hr.device)
            t_reshaped = rearrange(t, 'b -> b 1 1 1')
            x_t = t_reshaped * x_hr + (1 - t_reshaped) * x_noise
            u_t = x_hr - x_noise
            pred_u_t = model(x_t=x_t, time=t, x_cond=x_lr)

            loss = torch.mean((pred_u_t - u_t) ** 2) # mse
            loss.backward()
            optimizer.step()

            total_training_loss += loss.item()
            train_progress_bar.set_postfix(loss=f"{loss.item():.6f}")

        scheduler.step()
        mean_training_loss = total_training_loss / len(train_loader)
        train_loss_log.append(mean_training_loss)

        # validation
        model.eval()
        total_val_loss = 0.0

        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch}|Validation", leave=False, colour="#fb8500")
        with torch.no_grad():
            for batch in val_progress_bar:
                x_lr = batch['lr'].to(device)
                x_hr = batch['hr'].to(device)
                x_noise = (torch.rand_like(x_lr) * 2) - 1 # normalized to [-1, 1]
                
                t = torch.rand(x_lr.shape[0], device=x_lr.device)
                t_reshaped = rearrange(t, 'b -> b 1 1 1')
                
                x_t = t_reshaped * x_hr + (1 - t_reshaped) * x_noise
                u_t = x_hr - x_noise
                pred_u_t = model(x_t=x_t, time=t, x_cond=x_lr)

                loss = torch.mean((pred_u_t - u_t) ** 2) # mse
                total_val_loss += loss.item()
                val_progress_bar.set_postfix(loss=f"{loss.item():.6f}")

        mean_val_loss = total_val_loss / len(val_loader)
        val_loss_log.append(mean_val_loss)

        # live plot
        plot_live_losses(train_loss_log, val_loss_log)

        # save model
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
            }

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            torch.save(checkpoint, os.path.join(checkpoint_path, "best_model.pth"))
            print(f"Epoch {epoch}: New best model saved with validation loss: {mean_val_loss:.6f}")

    print(f"\n--- Training {num_params} params completed! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Conditional Flow Matching model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    
    parser.add_argument('--epochs', type=int, help='Override number of training epochs.')
    parser.add_argument('--learning_rate', type=float, help='Override learning rate.')
    parser.add_argument('--weight_decay', type=float, help='Override weight decay.')
    parser.add_argument('--batch_size', type=int, help='Override training batch size.')
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

    train_params = config['train_params']
    if args.epochs is not None:
        train_params['epochs'] = args.epochs
        print(f"Overriding train_params.epochs with: {args.epochs}")
    if args.learning_rate is not None:
        train_params['learning_rate'] = args.learning_rate
        print(f"Overriding train_params.learning_rate with: {args.learning_rate}")
    if args.weight_decay is not None:
        train_params['weight_decay'] = args.weight_decay
        print(f"Overriding train_params.weight_decay with: {args.weight_decay}")
    if args.batch_size is not None:
        train_params['batch_size'] = args.batch_size
        print(f"Overriding train_params.batch_size with: {args.batch_size}")
    
    train(config)