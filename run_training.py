from src.trainer import Trainer
from src.utils import load_config
import argparse


def main(args):
    config_path = args.config_path
    config = load_config(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='path to the model config')
    args = parser.parse_args()
    main(args)
