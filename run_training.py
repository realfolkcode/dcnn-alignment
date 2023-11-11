from src.trainer import Trainer
from src.utils import load_config
from src.asap import ASAPWrapper
from src.transforms import RandomJumps
from src.data import CrossSimilarityDataset
from src.dcnn import DCNN
import argparse

import wandb
import torch
from torchvision.transforms.v2 import Resize
from torch.utils.data import DataLoader
from torch.optim import Adam


def main(args):
    config_path = args.config_path
    project_name = args.project_name

    config = load_config(config_path)

    if project_name is not None:
        wandb.login()
        wandb.init(project=project_name,
                   config=config)
    
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
    trainer = Trainer(train_loader,
                      val_loader,
                      num_epochs,
                      optimizer,
                      device,
                      logging_fn=wandb.log)
    trainer.train(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='path to the model config')
    parser.add_argument('--project_name', type=str, required=False, default=None, help='wandb project name')
    args = parser.parse_args()
    main(args)
