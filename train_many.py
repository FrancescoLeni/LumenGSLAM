import subprocess
import json
from pathlib import Path
import copy
import os
import time
import torch
import numpy as np

json_path = Path(r'configs/train_configs')

with open(json_path / 'train_config.json', "r") as file:
    data_dict = json.load(file)

src = Path('configs/dataset_configs')

# for seq in os.listdir(src / 'yuyu_data'):
#     # if seq not in ["sigmoid_t2_a.json"]:
#     cmd = ["python", 'train.py'] + ['--train_config', f"train_config.json", '--dataset_config', f'yuyu_data/{seq}', '--name', f'gt_pose',
#                                     '--folder', f'prove', '--no_log']
#     subprocess.run(cmd)
#
# for seq in os.listdir(src / 'SCARED'):
#     # if seq not in ["cecum_t1_b.json"]:
#     cmd = ["python", 'train.py'] + ['--train_config', f"train_config.json", '--dataset_config', f'SCARED/{seq}', '--name', f'15_l2',
#                                     '--folder', f'every_5/after_tracking', '--no_log', '--tracking', '--no_eval']
# #     # cmd = ["python", 'eval.py'] + ['--gaussian_model', f"runs/train/C3VD/{seq.split('.')[0]}/every_5/just_pnp_init/parameters.pt", "--use_tracked_pose"]
# #
#     subprocess.run(cmd)





for seq in os.listdir(src / 'SCARED'):

    if seq == 'd1.json':
        continue

    if os.path.isdir(f"runs/train/SCARED/{seq.split('.')[0]}/constant_speed"):
        continue

    cmd = ["python", 'train.py'] + ['--train_config', f"constant_speed_scared.json", '--dataset_config', f"SCARED/{seq}",
                                    '--name', f'full_seq', '--folder', f'constant_speed', '--no_log',
                                    '--no_save_eval_imgs', '--tracking']

    # cmd = ["python", 'eval.py'] + ['--gaussian_model', f"runs/C3VD/{seq.split('.')[0]}/every_{j}/parameters.pt",
    #                                '--dataset_config', f'C3VD/{seq}', '--use_tracked_pose']

    subprocess.run(cmd)

    try:
        p_path = [x for x in Path(f"runs/train/SCARED/{seq.split('.')[0]}/constant_speed/full_seq").rglob('parameters*.pt')][0]
    except:
        print(f'seq {seq} not FAILED without a ckpt!')
        continue

    params = torch.load(p_path, map_location='cpu')

    trj = params['tracked_trj']

    pose_name = "constant_speed.txt"
    # Write to file
    txt = Path('dataset/SCARED') / seq.split('.')[0] / pose_name
    with open(txt, 'w') as f:
        for w2c in trj:
            w2c = w2c.numpy()

            c2w = np.linalg.inv(w2c)

            c2w = c2w.T  # to cope with C3VD being transposed for no reason
            flattened = c2w.flatten(order='C')
            line = ','.join(f"{v:.7g}" for v in flattened)

            f.write(line + '\n')

    cmd = ["python", 'train.py'] + ['--train_config', f"constant_speed_scared.json", '--dataset_config', f"SCARED/{seq}",
                                    '--name', f'split_seq', '--folder', f'constant_speed', '--no_log',
                                    '--no_save_eval_imgs', '--train_val_split', '--pose_file', pose_name]

    subprocess.run(cmd)


# new_data = copy.deepcopy(data_dict)
# new_data['use_every'] = 1
# new_data['do_phba'] = 'after_tracking'
#
# with open(json_path / f"use.json", "w") as file:
#     json.dump(new_data, file, indent=4)
#
# for seq in os.listdir(src / 'C3VD'):
#     cmd = ["python", 'train.py'] + ['--train_config', f"use.json", '--dataset_config', f"C3VD/{seq}",
#                                     '--name', f'pnp_phba_conf_2', '--folder', f'every_1/after_tracking', '--no_log',
#                                     '--tracking']
#     subprocess.run(cmd)
#
# os.remove(json_path / f"use.json")

# for j in [0.1, 0.5, 0.3]:
#     for k in [True, False]:
#         if (j == 0.1 and k == True):
#             new_data = copy.deepcopy(data_dict)
#
#             new_data['use_energy_mask'] = k
#             new_data['keyframes']['current_frame_prob'] = j
#             with open(json_path / f"use.json", "w") as file:
#                 json.dump(new_data, file, indent=4)
#
#             cmd = ["python", 'train.py'] + ['--train_config', f"use.json", '--dataset_config', f"Chiara_data/Colon_2_backward_reduced_100.json",
#                                             '--name', f'curr_{j}_mask_{k}', '--folder', f'prove/gt_pose', '--no_log']
#             subprocess.run(cmd)
#
#             os.remove(json_path / f"use.json")



# src_p = Path('runs/train/Chiara_data')
# folder = 'best'
# name = 'track'
#
# pose_name = 'my_tracked_pose.txt'
#
# dst = Path('configs/dataset_configs/Chiara_data')
#
# for seq in os.listdir(src_p):
#     if name is not None:
#         params_path = src_p / seq / folder / name / 'parameters.pt'
#     else:
#         params_path = src_p / seq / folder / 'parameters.pt'
#
#     params = torch.load(params_path, map_location='cpu')
#
#     trj = params['tracked_trj']
#
#     # Write to file
#     txt = dst / seq / pose_name
#     with open(txt, 'w') as f:
#         for w2c in trj:
#             w2c = w2c.numpy()
#
#             c2w = np.linalg.inv(w2c)
#
#             c2w = c2w.T  # to cope with C3VD being transposed for no reason
#             flattened = c2w.flatten(order='C')
#             line = ','.join(f"{v:.7g}" for v in flattened)
#
#             f.write(line + '\n')
#
# for seq in os.listdir(src):
#     # if seq not in ["sigmoid_t2_a.json"]:
#     cmd = ["python", 'train.py'] + ['--train_config', f"best.json", '--dataset_config', f'Chiara_data/{seq}', '--name', f'track_split',
#                                     '--folder', f'best', '--no_log', '--train_val_split', '--pose_file', f'{pose_name}']
#     subprocess.run(cmd)

# for seq in os.listdir(src):
#     # if seq not in ["sigmoid_t2_a.json"]:
#     cmd = ["python", 'eval.py'] + ['--gaussian_model', f"Other_models_exps/MonoGS/use_depth_loss/SLERP/{seq.split('.')[0]}/before_opt/parameters.pt",
#                                    '--dataset_config', f'C3VD/{seq}', '--train_val_split', '--pose_file', 'MonoGS_pose.txt']
#
#     subprocess.run(cmd)
    # cmd = ["python", 'eval.py'] + ['--gaussian_model', f"runs/train/C3VD/{seq.split('.')[0]}/train_val_split/EndoGSLAM_pose/parameters.pt",
    #                                '--dataset_config', f'C3VD/{seq}', '--pose_file', 'pose_EndoGSLAM.txt', '--train_val_split']
    # subprocess.run(cmd)
    #     cmd = ["python", 'eval.py'] + ['--gaussian_model', f"runs/train/C3VD/{seq.split('.')[0]}/train_val_split/tracked_pose/parameters.pt",
    #                                    '--dataset_config', f'C3VD/{seq}', '--train_val_split']
    #     subprocess.run(cmd)

# new_data = copy.deepcopy(data_dict)
#
# new_data['tracking']['ignore_reflexes'] = True
# with open(json_path / f"best.json", "w") as file:
#     json.dump(new_data, file, indent=4)
#
# for seq in os.listdir(src):
#         cmd = ["python", 'train.py'] + ['--train_config', f"best.json", '--dataset_config', f'C3VD/{seq}', '--name', f'pbr_only_diff_track',
#                                         '--folder', f'best', '--no_log', '--tracking']
#         subprocess.run(cmd)

# for seq in os.listdir(src):
#     # if seq not in ["cecum_t1_b.json"]:
#     cmd = ["python", 'train.py'] + ['--train_config', f"best.json", '--dataset_config', f'C3VD/{seq}', '--name', f'tracked_pre_dens',
#                                     '--folder', f'best', '--no_log', '--tracking']
#     subprocess.run(cmd)

# new_dict = copy.deepcopy(data_dict)
#
# new_dict['mapping']['loss']['use_normal_loss'] = True
#
# with open(json_path / f"use.json", "w") as file:
#     json.dump(new_dict, file, indent=4)
#
# for seq in os.listdir(src):
#
#     cmd = ["python", 'train.py'] + ['--train_config', f"use.json", '--dataset_config', f'C3VD/{seq}', '--name', f'norm_loss',
#                                     '--folder', f'sil_dens_camera', '--no_log']
#     subprocess.run(cmd)
#
# os.remove(json_path / f"use.json")
