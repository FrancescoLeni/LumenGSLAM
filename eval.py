import argparse

import torch
from pathlib import Path

from utils.training.loaders import EvalLoader, load_data_config
from utils.general import increment_path, my_logger, random_state
from utils.data_process.loading import load_params

from models.wrappers import Evaluer

from models.camera_model import Camera, SceneRenderer

TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'

# fixing random states
random_state()



def eval_loop(args):

    eval_dst = Path(str(Path(args.gaussian_model).parent).replace('/train/', '/test/'))

    if 'Other_models_exps' in str(eval_dst):
        eval_dst = eval_dst / 'eval'

    if args.name:
        eval_dst = eval_dst.parent / args.name

    eval_dst = increment_path(eval_dst.parent, eval_dst.name, exist_ok=args.exist_ok_dst)

    my_logger.info(f'saving to {eval_dst}')

    if args.device == 'cuda':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    dataset_config_path = Path('configs/dataset_configs') / args.dataset_config

    # load configs ...
    K, data_shape, depth_scale, data_path = load_data_config(dataset_config_path)

    # loading saved params
    gaussians, first_c2w, tracked_trj = load_params(args.gaussian_model)

    # to ensure that the eval is computed on tracked pose ONLY if it was tracked and the flag is True
    if args.use_tracked_pose and tracked_trj:
        pass
    else:
        tracked_trj = None

    # dataloader
    eval_loader = EvalLoader(data_path, data_shape, depth_scale, first_c2w=first_c2w, tracked_trj=tracked_trj,
                             pose_file_name=args.pose_file, split='val' if args.train_val_split else None,
                             use_every=args.use_every)

    # model
    camera = Camera(w=data_shape[1], h=data_shape[0], k=K, device=device, near=0.01, far=100, bg=[0,0,0])

    renderer = SceneRenderer(device=device)

    evaluer = Evaluer(camera, gaussians, eval_loader, renderer, device, eval_dst, save_images=not args.no_save_imgs)

    evaluer.eval_save()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gaussian_model', type=str, required=True, help='path to gaussian model params')
    parser.add_argument("--dataset_config", type=str, required=True, help="name of dataset config json (in config/")
    parser.add_argument("--device", type=str, default="cuda", help="device to use")
    parser.add_argument('--name', default=None, type=str, help='name to use')
    parser.add_argument("--pose_file", type=str, default='pose.txt', help="name of pose file to use")
    parser.add_argument("--train_val_split", action="store_true", help='whether to use train val split')
    parser.add_argument('--use_every', default=1, type=int, help='how to sample the available frames')
    parser.add_argument("--no_save_imgs", action='store_true', help="whether not to save images during eval")
    parser.add_argument("--exist_ok_dst", action='store_true', help="whether to overwrite existing dst")

    parser.add_argument('--use_tracked_pose', action='store_true', help="whether to use tracked pose "
                                     "(N.B. only works if the posed was tracked during training, otherwise will use gt")

    args = parser.parse_args()

    eval_loop(args)