import argparse
import os
import gan 

class GenerateFruitsData():

    
    def generate_fruits_data():
        predictions = generator(test_input, training=False)













if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate Fruit Images.',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-c', '--class', help='Resume checkpoint path', required=True)
    parser.add_argument('--images-per-class', help='Number of images per class to output', required=True, default=10, type=int)

    args = parser.parse_args()

    GenerateFruitsData().generate_fruits_data(**vars(args))
