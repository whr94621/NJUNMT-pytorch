import argparse
from src.main import train

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str,
                    help="The name of the model. Will alse be the prefix of saving archives.")
parser.add_argument('--reload', action="store_true",
                    help="Whether to restore from the latest archives.")
parser.add_argument('--config_path', type=str,
                    help="The path to config file.")
parser.add_argument('--log_path', type=str,
                    help="The path for saving tensorboard logs. Default is ./log")
parser.add_argument('--saveto', type=str,
                    help="The path for saving models. Default is ./save.")

parser.add_argument('--debug', action="store_true",
                    help="Use debug mode.")

parser.add_argument('--use_gpu', action="store_true",
                    help="Whether to use GPU.")

if __name__ == '__main__':

    args = parser.parse_args()
    train(args)