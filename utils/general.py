import os
import numpy as np
import random
import warnings
import torch
from pathlib import Path
import json

import logging


def init_logger():

    logger = logging.getLogger("my_new_logger")

    # Set the level
    logger.setLevel(logging.INFO)

    # Create a handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter("")
    handler.setFormatter(formatter)

    # Add the handler to your logger
    logger.addHandler(handler)

    # Disable propagation to the root logger
    logger.propagate = False

    # if not logger.handlers:
    #     logger.addHandler(handler)

    return logger


# GLOBAL CALL
my_logger = init_logger()


# fixes random states to same seed and silenced warnings
def random_state(seed=36):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['MPLCONFIGDIR'] = os.getcwd() + '/configs/'

    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)

    np.random.seed(seed)
    # _ = np.random.default_rng(seed)

    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)


def integer_division(a: int, b: int, floor=False):
    """

    args:
        -a: first term
        -b: second term
        -floor: if true returns division "per difetto", False returns "per eccesso"
    :return:
    """

    if floor:
        return a // b
    else:
        return a // b + 1


def increment_path(folder="runs", name="exp", exist_ok=False, sep=''):
    """

    Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.

    :param
        folder: addressed folder
        name: name of the new directory inside folder
        exist_ok: if True keeps the existing dir, else actually increments it
        sep: how to connect the increment with name (see above)
    """

    path = Path(folder) / name  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)
    if not os.path.isdir(path):
        os.makedirs(path)

    my_logger.info(f"saving folder is {path}")
    return path

def json_from_parser(parser_args, save_path, name="arguments.json"):
    args_dict = vars(parser_args)

    save_path = save_path / name
    with open(save_path, 'w') as json_file:
        json.dump(args_dict, json_file, indent=2)

def min_max(x):
    max = x.max()
    min = x.min()

    return (x - min) / (max - min)

def points_to_mask_torch(rounded_points, mask_shape):
    mask = torch.zeros(mask_shape, dtype=torch.uint8, device=rounded_points.device)

    H, W = mask_shape

    # Split x and y
    x = rounded_points[:, 0].long()
    y = rounded_points[:, 1].long()

    # Filter valid indices (within image bounds)
    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x = x[valid]
    y = y[valid]

    # Set valid positions to 1
    mask[y, x] = 1
    return mask