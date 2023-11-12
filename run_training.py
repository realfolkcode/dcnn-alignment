from src.trainer import Trainer
from src.utils import load_config
from src.asap import ASAPWrapper
from src.transforms import RandomJumps
from src.data import CrossSimilarityDataset
from src.dcnn import DCNN
import argparse
from functools import partial

import wandb
import torch
from torchvision.transforms.v2 import Resize
from torch.utils.data import DataLoader
from torch.optim import Adam


def log_images(images, y_pred, table):
    log_images = images.cpu().numpy()
    log_y_pred = y_pred.cpu().numpy()
    bs = len(log_images)
    for i in range(min(4, bs)):
        table.add_data(wandb.Image(log_images[i]), log_y_pred[i])
    wandb.log({"val_predictions" : table})


def main(args):
    config_path = args.config_path
    project_name = args.project_name

    config = load_config(config_path)

    if project_name is not None:
        wandb.login()
        wandb.init(project=project_name,
                   config=config)
        columns=["image", "y_pred"]
        val_table = wandb.Table(columns=columns)
    
    asap_dir = config['data']['asap_dir']
    val_ratio = config['data']['val_ratio']
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']

    img_size = config['transforms']['img_size']
    fs = config['transforms']['fs']
    min_num_jumps = config['transforms']['min_num_jumps']
    max_num_jumps = config['transforms']['max_num_jumps']
    max_silence_s = config['transforms']['max_silence_s']

    hidden_channels = config['model']['hidden_channels']

    num_epochs = config['train']['num_epochs']
    lr = config['train']['lr']
    weight_decay = config['train']['weight_decay']
    device = torch.device(config['train']['device'])

    asap_wrapper = ASAPWrapper(asap_dir, val_ratio=val_ratio, random_seed=42)

    transform = Resize((img_size, img_size))
    jumps_transform = RandomJumps(fs, 
                                  min_num_jumps=min_num_jumps, 
                                  max_num_jumps=max_num_jumps, 
                                  max_silence_s=max_silence_s)

    train_dataset = CrossSimilarityDataset(asap_wrapper.train_paths,
                                           fs,
                                           transform,
                                           structural_transform=jumps_transform,
                                           inference_only=False)
    val_dataset = CrossSimilarityDataset(asap_wrapper.val_paths,
                                         fs,
                                         transform,
                                         structural_transform=jumps_transform,
                                         inference_only=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    model = DCNN(img_size, hidden_channels, max_num_jumps * 2).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if project_name is not None:
        metrics_logger = wandb.log
        images_logger = partial(log_images, table=val_table)
    else:
        metrics_logger = None
        images_logger = None

    trainer = Trainer(train_loader,
                      val_loader,
                      num_epochs,
                      optimizer,
                      device,
                      metrics_logger=metrics_logger,
                      images_logger=images_logger)
    
    trainer.train(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='path to the model config')
    parser.add_argument('--project_name', type=str, required=False, default=None, help='wandb project name')
    args = parser.parse_args()
    main(args)
