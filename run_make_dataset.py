from src.asap import ASAPWrapper
from src.data import make_dataset
import argparse


def main(args):
    asap_dir = args.asap_dir
    train_dir = args.train_dir
    val_dir = args.val_dir
    val_ratio = args.val_ratio
    fs = args.fs

    asap_wrapper = ASAPWrapper(asap_dir, val_ratio=val_ratio, random_seed=42)
    make_dataset(train_dir, asap_wrapper.train_paths, fs=fs)
    make_dataset(val_dir, asap_wrapper.val_paths, fs=fs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--asap_dir', type=str, required=True, help='path to ASAP directory')
    parser.add_argument('--train_dir', type=str, required=True, help='path to training directory')
    parser.add_argument('--val_dir', type=str, required=True, help='path validation directory')
    parser.add_argument('--val_ratio', type=float, required=False, default=0.2, help='validation set ratio')
    parser.add_argument('--fs', type=int, required=False, default=10, help='piano roll sampling frequency')
    args = parser.parse_args()
    main(args)
