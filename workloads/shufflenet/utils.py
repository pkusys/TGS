from easydict import EasyDict as edict
import json
import argparse
import os
import tensorflow as tf
from pprint import pprint
import sys
import socket
import logging

def parse_args():
    """
    Parse the arguments of the program
    :return: (config_args)
    :rtype: tuple
    """
    # Create a parser
    parser = argparse.ArgumentParser(description="ShuffleNet TensorFlow Implementation",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')
    parser.add_argument('--config', default=None, type=str, help='Configuration file')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training')
    parser.add_argument('--num-epochs', type=int, default=1000)

    # Parse the arguments
    args = parser.parse_args()

    # Parse the configurations from the config json file provided
    try:
        if args.config is not None:
            with open(args.config, 'r') as config_file:
                config_args_dict = json.load(config_file)
        else:
            print("Add a config file using \'--config file_name.json\'", file=sys.stderr)
            exit(1)

    except FileNotFoundError:
        print("ERROR: Config file not found: {}".format(args.config), file=sys.stderr)
        exit(1)
    except json.decoder.JSONDecodeError:
        print("ERROR: Config file is not a proper JSON file!", file=sys.stderr)
        exit(1)

    config_args = edict(config_args_dict)

    config_args.num_epochs = args.num_epochs
    config_args.batch_size = args.batch_size
    pprint(config_args)
    print("\n")

    return config_args


def create_experiment_dirs(exp_dir):
    """
    Create Directories of a regular tensorflow experiment directory
    :param exp_dir:
    :return summary_dir, checkpoint_dir:
    """
    experiment_dir = os.path.realpath(os.path.join(os.path.dirname(__file__))) + "/experiments/" + exp_dir + "/"
    summary_dir = experiment_dir + 'summaries/'
    checkpoint_dir = experiment_dir + 'checkpoints/'
    # output_dir = experiment_dir + 'output/'
    # test_dir = experiment_dir + 'test/'
    # dirs = [summary_dir, checkpoint_dir, output_dir, test_dir]
    dirs = [summary_dir, checkpoint_dir]
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        print("Experiment directories created!")
        # return experiment_dir, summary_dir, checkpoint_dir, output_dir, test_dir
        return experiment_dir, summary_dir, checkpoint_dir
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def calculate_flops():
    # Print to stdout an analysis of the number of floating point operations in the
    # model broken down by individual operations.
    tf.profiler.profile(
        tf.get_default_graph(),
        options=tf.profiler.ProfileOptionBuilder.float_operation(), cmd='scope')


def show_parameters():
    tf.profiler.profile(
        tf.get_default_graph(),
        options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter(), cmd='scope')


def load_obj(filename):
    import pickle
    with open(filename, 'rb') as file:
        return pickle.load(file)


def make_logger(name):
    LOG_FORMAT = '{name}:{levelname} [{asctime}] {message}'

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(LOG_FORMAT, style='{'))
    logger.addHandler(ch)

    return logger


def get_host_ip():
    """get the host ip elegantly
    https://www.chenyudong.com/archives/python-get-local-ip-graceful.html
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip