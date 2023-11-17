from src.trainer import Trainer
from src.utils import load_config, plot_cross_similarity
from src.transforms import RandomJumps
from src.data import CrossSimilarityDataset
from src.dcnn import DCNN
import argparse
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter('ignore')

import wandb
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def log_images(images, y_pred):
    log_images = images.cpu().numpy()
    log_y_pred = y_pred.cpu().numpy()
    bs = len(log_images)
    for i in range(min(4, bs)):
        plot_cross_similarity(log_images[i], inflection_points=log_y_pred[i])
        wandb.log({"val_predictions": plt})


def main(args):
    config_path = args.config_path
    checkpoint_path = args.checkpoint_path
    project_name = args.project_name

    config = load_config(config_path)

    if project_name is not None:
        wandb.login()
        wandb.init(project=project_name,
                   config=config)
    
    train_dir = config['data']['train_dir']
    val_dir = config['data']['val_dir']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']

    img_size = config['transforms']['img_size']
    fs = config['transforms']['fs']
    pix_mean = config['transforms']['pix_mean']
    pix_std = config['transforms']['pix_std']
    min_num_jumps = config['transforms']['min_num_jumps']
    max_num_jumps = config['transforms']['max_num_jumps']
    max_silence_s = config['transforms']['max_silence_s']

    hidden_channels = config['model']['hidden_channels']

    num_epochs = config['train']['num_epochs']
    lr = config['train']['lr']
    weight_decay = config['train']['weight_decay']
    device = torch.device(config['train']['device'])

    transform = v2.Compose([v2.Resize((img_size, img_size), antialias=True),
                            v2.Normalize(mean=[pix_mean], std=[pix_std])])
    jumps_transform = RandomJumps(fs, 
                                  min_num_jumps=min_num_jumps, 
                                  max_num_jumps=max_num_jumps, 
                                  max_silence_s=max_silence_s)

    train_dataset = CrossSimilarityDataset(train_dir,
                                           transform,
                                           structural_transform=jumps_transform,
                                           inference_only=False)
    val_dataset = CrossSimilarityDataset(val_dir,
                                         transform,
                                         structural_transform=jumps_transform,
                                         inference_only=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    model = DCNN(img_size, hidden_channels, max_num_jumps * 2).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    if project_name is not None:
        metrics_logger = wandb.log
        images_logger = log_images
    else:
        metrics_logger = None
        images_logger = None

    trainer = Trainer(train_loader,
                      val_loader,
                      num_epochs,
                      optimizer,
                      device,
                      checkpoint_path,
                      metrics_logger=metrics_logger,
                      images_logger=images_logger,
                      scheduler=scheduler)
    
    trainer.train(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='path to the model config')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='checkpoint path')
    parser.add_argument('--project_name', type=str, required=False, default=None, help='wandb project name')
    args = parser.parse_args()
    main(args)
