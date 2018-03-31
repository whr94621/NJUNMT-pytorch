import argparse
from src.main import train

parser = argparse.ArgumentParser()

parser.add_argument("--data", type=str)
parser.add_argument('--model_name', type=str)
parser.add_argument('--reload', action="store_true")
parser.add_argument('--config_path', type=str)
parser.add_argument('--log_path', type=str)
parser.add_argument('--pretrain_path', type=str, default="")
parser.add_argument('--saveto', type=str)
parser.add_argument('--debug', action="store_true")
parser.add_argument('--use_gpu', action="store_true")

if __name__ == '__main__':

    args = parser.parse_args()
    train(args)