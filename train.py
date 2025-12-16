import argparse
import os
import shutil

import torch
from pathlib import Path


from utils.training.loaders import BaseDataLoader, load_data_config, load_config_json, EvalLoader
from utils.training.logger import TrainLogger
from utils.general import increment_path, my_logger, random_state, json_from_parser
from utils.data_process.loading import load_params


from models.wrappers import Trainer, Evaluer

from models.camera_model import Camera, SceneRenderer
from models.gaussian_model import GaussianModel

TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'

# fixing random states
random_state(0)


def train_loop(args):


    if args.train_val_split:
        train_split = 'train'
        val_split = 'val'
    else:
        train_split = None
        val_split = None


    # load configs ...
    train_config_path = Path('configs/train_configs') / args.train_config
    dataset_config_path = Path('configs/dataset_configs') / args.dataset_config

    train_config = load_config_json(train_config_path) # gaussian config, densification config, tracking, mapping
    K, data_shape, depth_scale, data_path = load_data_config(dataset_config_path)


    # setting saving dir
    train_dst = Path('runs/train') / data_path.parent.name / data_path.name

    if args.folder:
        train_dst = train_dst / args.folder

    os.makedirs(train_dst, exist_ok=True)
    train_dst = increment_path(train_dst, args.name)

    # copying configs to saving dir
    shutil.copy(train_config_path, train_dst / 'train_config.json')
    shutil.copy(dataset_config_path, train_dst)

    # saving parser args
    json_from_parser(args, train_dst)

    if args.tracking:
        tracking = True
    else:
        tracking = False

    if args.device == 'cuda':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # dataloader
    train_loader = BaseDataLoader(data_path, data_shape, depth_scale, pose_file_name=args.pose_file, split=train_split)

    # model
    camera = Camera(w=data_shape[1], h=data_shape[0], k=K, device=device, near=0.01, far=100, bg=[0,0,0]) # should put far = 1
    gaussian_population = GaussianModel(device=device, config=train_config, save_dst=Path(train_dst))

    renderer = SceneRenderer(device=device)

    logger = TrainLogger(train_dst, (args.tracking and train_config['do_phba'] in ['after_tracking', 'after_expansion', 'after_mapping']))

    # wrapper of model for train
    trainer = Trainer(camera, gaussian_population, train_loader, renderer, logger, tracking, device, train_config, not (args.no_map_metrics or args.no_log))

    # run train loop
    trainer.train_loop()

    # saving parameters
    trainer.save_checkpoint("final")

    # saving times
    trainer.timer.save_info(train_dst)

    # plots and stats
    if not args.no_log:
        logger.plot_save(plot_metrics=not args.no_map_metrics)

    logger.save_info_csv()
    trainer.save_keyframe_stats()

    # eval loop
    if not args.no_eval:
        eval_dst = Path(str(train_dst).replace('train', 'test'))
        os.makedirs(eval_dst, exist_ok=True)

        eval_data_src = data_path

        my_logger.info(f'starting evaluation on: {eval_data_src}')

        gaussians, first_c2w, tracked_trj = load_params(train_dst / 'parameters.pt')

        eval_loader = EvalLoader(eval_data_src, data_shape, depth_scale, first_c2w=first_c2w, tracked_trj=tracked_trj,
                                 pose_file_name=args.pose_file, split=val_split, use_every=train_config['use_every'])

        evaluer = Evaluer(camera, gaussians, eval_loader, renderer, device, eval_dst, save_images=not args.no_save_eval_imgs)

        evaluer.eval_save()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str, required=True, help="name of training_config json")
    parser.add_argument("--dataset_config", type=str, required=True, help="name of dataset config json")
    parser.add_argument("--device", type=str, default="cuda", help="device to use")
    parser.add_argument("--folder", type=str, default=None, help="name of the experiment folder")
    parser.add_argument("--name", type=str, default='exp', help="name of the experiment folder")
    parser.add_argument("--no_eval", action='store_true', help="whether not to eval")
    parser.add_argument("--no_save_eval_imgs", action='store_true', help="whether not to save images during eval")
    parser.add_argument("--no_map_metrics", action='store_true', help="whether not to compute map training metrics")
    parser.add_argument("--no_log", action='store_true', help="whether to not compute and log metrics and loss during training")
    parser.add_argument("--pose_file", type=str, default='pose.txt', help="name of pose file to use")
    parser.add_argument("--train_val_split", action='store_true', help="whether to split between train and val")

    parser.add_argument("--tracking", action="store_true", help="whether to do tracking and estimate camera pose")

    args = parser.parse_args()

    train_loop(args)