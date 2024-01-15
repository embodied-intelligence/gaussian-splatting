#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
from argparse import ArgumentParser

import torch
import numpy as np
import subprocess

import torch
from matplotlib import pyplot as plt

from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel


cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
os.system('echo $CUDA_VISIBLE_DEVICES')

from scene import Scene
from gaussian_renderer import render
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams
from gaussian_renderer import GaussianModel

def run_render():
    from argparse import ArgumentParser, Namespace
    parser = ArgumentParser(description="Testing script parameters")

    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    #load a model
    cmdlne_string = ["-m", "/home/beantown/ran/gaussian-splatting/output/5db04bc0-9"]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
    print("Looking for config file in", cfgfilepath)
    with open(cfgfilepath) as cfg_file:
        print("Config file found: {}".format(cfgfilepath))
        cfgfile_string = cfg_file.read()

    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    args = Namespace(**merged_dict)

    #Initialize system state (RNG)
    safe_state(args.quiet)

    ##render_sets
    dataset = model.extract(args)
    iteration = args.iteration
    pipeline = pipeline.extract(args)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    return (dataset.model_path, scene.loaded_iter, scene.getTrainCameras()[0], gaussians, pipeline, background)

def transformation_vuer2gs(matrix):
    transf = np.array(matrix).reshape(4, 4)

    Rt = np.zeros((4, 4))
    Rt[:3, :3] = transf[:3, :3].transpose()
    Rt[:3, 3] = transf[:3, 3]
    Rt[3, 3] = 1.0

    # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    Rt[:3, 1:3] *= -1  # 0,-1,0 up
    R = Rt[:3, :3]
    T = Rt[:3, 3]
    return (R,T)

def run_render_with_view(view, others):
    #render_set
    from scene import cameras
    import math
    model_path, loaded_iter, test_camera, gaussians, pipeline, background = others

    with torch.no_grad():
        view = view['camera']
        gt = test_camera.original_image[0:3, :, :]

        #Rt = c2w
        R, T = transformation_vuer2gs(view['matrix'])
        fov_rad = math.radians(view['fov'])
        trans = np.array(view['position'])

        gs_camera = cameras.Camera(
            colmap_id=0, R=R, T=T,
            FoVx=fov_rad, FoVy=fov_rad,
            image= gt, gt_alpha_mask=None,
            image_name=None, uid=0,
            trans=trans
        )

        rendering = render(gs_camera, gaussians, pipeline, background)["render"]
        ndarr = rendering.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    return ndarr.astype(np.uint8)


if __name__ == "__main__":
    # Set up command line argument parser
    # parser = ArgumentParser(description="Testing script parameters")
    # model = ModelParams(parser, sentinel=True)
    # pipeline = PipelineParams(parser)
    # parser.add_argument("--iteration", default=-1, type=int)
    # parser.add_argument("--skip_train", action="store_true")
    # parser.add_argument("--skip_test", action="store_true")
    # parser.add_argument("--quiet", action="store_true")
    # args = get_combined_args(parser)
    # print("Rendering " + args.model_path)
    #
    # # Initialize system state (RNG)
    # safe_state(args.quiet)
    #
    # render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
    view  = {'camera': {
    'matrix': [1.0, 0.0, 0.0, 0.0,
               0.0, 1.0, 0.0, 0.0,
               0.0, 0.0, 1.0, 0.0,
               0.0, 0.0, 0.0, 1.0],
    'position': [0.0, 0.0, 0.0],
    'fov': 75,
    },
}
    others = run_render()
    run_render_with_view(view, others)