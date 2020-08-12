import argparse
import os
import sys 
from src.train import Train

class Config:
    def __init__(self, **kwargs):
        for argument, value in kwargs.items():
            setattr(self, argument, value)

    def __repr__(self):
        repr(vars(self))

    def __str__(self):
        repr(vars(self))

# ---------------------------------------------------------------------------- #


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train Fruit GAN Model.',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--dataset-path', help='Training dataset', default='cifar10')
    parser.add_argument('--batch-size', help='Specifies resolution', default=None, type=int)
    parser.add_argument('--total-train-examples', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    parser.add_argument('--DiffAugment', help='Comma-separated list of DiffAugment policy', default='color,cutout')
    parser.add_argument('--epochs', help='Batch size', default=None, type=int, metavar='N')
    parser.add_argument('--learning-rate', help='Training length in thousands of images (default: %(default)s)', metavar='KIMG', default=25000, type=int)
    parser.add_argument('--noise-dim', help='Half-life of exponential moving average in thousands of images', metavar='KIMG', default=1000, type=int)
    parser.add_argument('--image-size', help='Number of samples', default=None, type=int)
    parser.add_argument('--batch-size', help='Number of feature maps', default=2048, type=int)
    parser.add_argument('--seed', help='Maximum number of feature maps', default=None, type=int)
    parser.add_argument('--model', help='Latent size', default=None, type=int)
    parser.add_argument('--loss', help='Mirror augment (default: %(default)s)', default=True, metavar='BOOL', type=_str_to_bool)
    parser.add_argument('--optimizer', help='Custom op implementation (default: %(default)s)', default='cuda')


    args = parser.parse_args()

    for metric in args.metrics:
        if metric not in metric_defaults:
            print('Error: unknown metric \'%s\'' % metric)
            sys.exit(1)
    
    config = Config(**vars(args))
    train = Train()
    train(config)


