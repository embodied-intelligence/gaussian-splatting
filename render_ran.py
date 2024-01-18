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
import numpy as np
import subprocess

import torch
from scene import Scene
from gaussian_renderer import render
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams
from gaussian_renderer import GaussianModel
from vuer_utils_se3 import rotation_matrix

cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
os.system('echo $CUDA_VISIBLE_DEVICES')


def GSArgs(model_path):
    from argparse import ArgumentParser, Namespace
    parser = ArgumentParser(description="Testing script parameters")
    model_args = ModelParams(parser, sentinel=True)
    pipline_args = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    cmdlne_string = ["-m", model_path]
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
    return args, model_args, pipline_args

def main():

    model_path = "/home/beantown/ran/gaussian-splatting/output/fd986845-9" # ball_add
    #model_path = "/home/beantown/ran/gaussian-splatting/output/353df241-2" # stair_feature1
    model_path = "/home/beantown/datasets/lucidSim/scenes/mit_stairs/stairs_0004_v1/output_gs"
    args,  model_args, pipline_args= GSArgs(model_path=model_path)

    #Initialize system state (RNG)
    safe_state(args.quiet)

    ##render_sets
    dataset = model_args.extract(args)
    iteration = args.iteration
    pipeline = pipline_args.extract(args)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        gt = scene.getTrainCameras()[0].original_image[0:3, :, :]
        h, w = gt.size()[1:]
        aspect_ratio = w / h
    return (dataset.model_path, scene.loaded_iter, (w, h, aspect_ratio), gaussians, pipeline, background)

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

def get_dic2arr(data):
    key_list = list(data.keys())
    return np.array([data[key] for key in key_list]), key_list


def run_render_with_view(view, others):
    #render_set
    from scene import cameras
    import math
    model_path, loaded_iter, ori_render_img, gaussians, pipeline, background = others
    ori_w, ori_h, ori_aspect = ori_render_img
    with (torch.no_grad()):
        world = view['world']
        view= view['camera']

        #Rt = c2w
        #world = world_cordinate, camera-local coordinate
        (wx, wy, wz), _ = get_dic2arr(world["rotation"])
        world_mat = rotation_matrix(wx, wy, wz, unit="deg")
        world_trans, _ = get_dic2arr(world["position"])

        R, T = transformation_vuer2gs(view['matrix'])
        R = world_mat@R
        T += world_trans
        fov_rad = math.radians(view['fov'])

        #world_trans, _ = get_dic2arr(world["position"])
        trans = np.array(view['position']) #+ world_trans

        camera_height = view["height"]
        camera_width = view["width"]
        scale =  world["scale"]

        if view["aspect"] <= ori_aspect:
            render_height = camera_height
            render_width = ori_w * camera_height / ori_h
            crop_w, crop_h = int((render_width-camera_width)/2), 0
        else:
            render_height = camera_width / ori_w * ori_h
            render_width = camera_width
            crop_w, crop_h = 0, int((render_height - camera_height) / 2),

        gs_camera = cameras.Camera(
            colmap_id=0, R=R, T=T,
            FoVx=fov_rad, FoVy=fov_rad,
            image= None, gt_alpha_mask=None,
            image_name=None, uid=0,
            trans=trans, scale = scale,
            image_width=render_width, image_height = render_height,
        )

        rendering = render(gs_camera, gaussians, pipeline, background)["render"]
        ndarr = rendering.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        ndarr = ndarr[crop_w:crop_w+camera_width, crop_h:crop_h+camera_height]

    return ndarr.astype(np.uint8)


if __name__ == "__main__":

    view  = {'camera': {
    'matrix': [1.0, 0.0, 0.0, 0.0,
               0.0, 1.0, 0.0, 0.0,
               0.0, 0.0, 1.0, 0.0,
               0.0, 0.0, 0.0, 1.0],
    'position': [0.0, 0.0, 0.0],
    'width': 1024,
    'height': 768,
    'fov': 75,
        },
    }
    others = main()
    run_render_with_view(view, others)