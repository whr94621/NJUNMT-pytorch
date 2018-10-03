import argparse
from src.main import train
from . import auto_mkdir

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str,
                    help="The name of the model. Will alse be the prefix of saving archives.")

parser.add_argument('--reload', action="store_true",
                    help="Whether to restore from the latest archives.")

parser.add_argument('--config_path', type=str,
                    help="The path to config file.")

parser.add_argument('--log_path', type=str, default="./log",
                    help="The path for saving tensorboard logs. Default is ./log")

parser.add_argument('--saveto', type=str, default="./save",
                    help="The path for saving models. Default is ./save.")

parser.add_argument('--debug', action="store_true",
                    help="Use debug mode.")

parser.add_argument('--use_gpu', action="store_true",
                    help="Whether to use GPU.")

parser.add_argument('--pretrain_path', type=str, default="", help="The path for pretrained model.")

parser.add_argument("--valid_path", type=str, default="./valid",
                    help="""Path to save translation for bleu evaulation. Default is ./valid.""")


def run(**kwargs):
    args = parser.parse_args()

    # Modify some options.
    for k, v in kwargs.items():
        setattr(args, k, v)

    auto_mkdir(args.log_path)
    auto_mkdir(args.saveto)
    auto_mkdir(args.valid_path)

    train(args)


if __name__ == '__main__':
    run()
