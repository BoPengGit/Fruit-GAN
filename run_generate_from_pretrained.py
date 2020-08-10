import argparse
import os
import tensorflow as tf 
import gan 


def generate_images(generator_model, discriminator_model, output_dir, num_images_to_generate):
    seed = tf.random.normal([num_images_to_generate, noise_dim])
    predictions = model(seed, training=False)

# ---------------------------------------------------------------------------- #


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate Fruit Images from pre-trained GAN model.',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--generator-model', help='Pretrained Fruit GAN Generator model file location', type=str)
    parser.add_argument('--discriminator-model', help='Pretrained Fruit GAN Discriminator model file location', type=str)
    parser.add_argument('--output_dir', help='Directory path of output generated fruit image files', default='data/generated_fruits/', type=str)
    parser.add_argument('--num-images-to-generate', help='Number count of desired output generated images', default=100, type=int)

    args = parser.parse_args()

    for metric in args.metrics:
        if metric not in metric_defaults:
            print('Error: unknown metric \'%s\'' % metric)
            sys.exit(1)
    
    run(**vars(args))

