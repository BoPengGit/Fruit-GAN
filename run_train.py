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

    parser.add_argument('--dataset-path', help='Dataset Path', default='gs://ai_evaluation/Fruits/tfrecords/train/*.tfrecords')
    parser.add_argument('--total-train-examples', help='Total number of examples/images in training dataset', default=30243, type=int)
    parser.add_argument('--DiffAugment', help='Comma-separated list of DiffAugment policy', default='horizontal_flip,color,translation,cutout')
    parser.add_argument('--epochs', help='Number of Training Epochs', default=100, type=int, )
    parser.add_argument('--learning-rate', help='Learning Rate of Optimizer', default=3e-4, type=int)
    parser.add_argument('--noise-dim', help='Length of GAN noise array', default=128, type=int)
    parser.add_argument('--image-size', help='Size of input image', default=98, type=int)
    parser.add_argument('--batch-size', help='Global Batch Size', default=32, type=int)
    parser.add_argument('--model', help='GAN model Name', default='DCGAN', type=str)
    parser.add_argument('--loss', help='GAN ML loss name', default='GANCrossEntropyLoss', type=str)
    parser.add_argument('--optimizer', help='GAN optimizer name', default='Adam', type=str)


    args = parser.parse_args()

    for metric in args.metrics:
        if metric not in metric_defaults:
            print('Error: unknown metric \'%s\'' % metric)
            sys.exit(1)
    
    config = Config(**vars(args))
    train = Train()
    train(config)


