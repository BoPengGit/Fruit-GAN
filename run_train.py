import argparse
import os
import gan 



# ---------------------------------------------------------------------------- #



dataset_path
batch_size                  = 32 # Choose your own optimal batch size
total_train_examples        = 30243 # Number of training examples
steps_per_epoch             = total_train_examples // batch_size # Number of training steps per epoch
per_replica_batch_size      = batch_size // strategy.num_replicas_in_sync # batch size per tpu replica
EPOCHS                      = 5 # Choose the number of traning epochs
learning_rate               = 0.0001 # Choose the learning rate 
noise_dim                   = 128 # Choose dimension of the noise vector
num_examples_to_generate    = 32 # Number of examples to generate after each epoch
image_size                  = 98 # W and H of the image, for even faster training you can choose to crop it to smaller sizes
num_channels                = 3 # Number of color channels (RGB -> 3)
BATCH_SIZE = 32
### Setting the seed is going to help with tracking the generator's progress, no need to change!
seed                        = tf.random.normal([num_examples_to_generate, noise_dim])
model 
loss 
optimizer 



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train Fruit GAN Model.',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--dataset-path', help='Training dataset', default='cifar10')
    parser.add_argument('--batch-size', help='Specifies resolution', default=None, type=int)
    parser.add_argument('--total-train-examples', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    parser.add_argument('--DiffAugment', help='Comma-separated list of DiffAugment policy', default='color,cutout')
    parser.add_argument('--steps-per-epoch', help='Number of GPUs (default: %(default)s)', default=1, type=int, metavar='N')
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
    
    run(**vars(args))

