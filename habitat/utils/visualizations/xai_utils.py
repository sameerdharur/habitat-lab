#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import textwrap
from typing import Dict, List, Optional, Tuple
import pdb
import imageio
import numpy as np
import cv2
import matplotlib.cm
import matplotlib
import tqdm
from PIL import Image, ImageDraw
import torch
from habitat.core.logging import logger
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import draw_collision, images_to_video
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
cv2 = try_cv2_import()


def compute_grad_cam(values, actions, gradcam_layer, egocentric_view, envs, device):
    action_one_hot = torch.zeros_like(values)
    action_one_hot[range(envs), actions] = 1
    gradients = torch.autograd.grad(outputs=values, inputs=gradcam_layer, grad_outputs=action_one_hot, create_graph=True)[0].to(device)
    #gradients = torch.autograd.grad(outputs=values, inputs=batch['rgb'], grad_outputs=action_one_hot, create_graph=True)[0].to(self.device)
    gradients = gradients.detach().cpu().numpy()
    weights = np.mean(gradients, axis=(2, 3))
    activations = gradcam_layer.detach().cpu().numpy()
    cam = np.zeros((activations.shape[0],activations.shape[2],activations.shape[3]), dtype=np.float32)
    for j in range(cam.shape[0]):
        for i, w in enumerate(weights[j]):
            cam[j] += w * activations[j,i, :, :]
    cam = np.maximum(cam, 0)
    torch.cuda.empty_cache()
    cam_scaled = np.array(Image.fromarray(cam[0].astype(np.float)).resize(egocentric_view[:,:,0].shape))
    cam1 = cam_scaled
    cam2 = cam1 - np.min(cam1)
    cam3 = cam2/(np.max(cam2))
    img1 = egocentric_view + (matplotlib.cm.jet(cam3)[:,:,:3]*255)
    img2 = img1 / np.max(img1)
    img2 *= 255
    return img2

def compute_saliency_maps(values, actions, rgb_inputs, egocentric_view, envs, device):
    action_one_hot = torch.zeros_like(values)
    action_one_hot[range(envs), actions] = 1
    gradients = torch.autograd.grad(outputs=values, inputs=rgb_inputs, grad_outputs=action_one_hot, create_graph=True)[0].to(device)
    gradients = gradients.squeeze()
    #saliency, _ = torch.max(gradients, 2) # For RGB input
    saliency = gradients # For Depth input
    saliency = saliency.detach().cpu().numpy()
    torch.cuda.empty_cache()
    saliency = saliency * 10000
    img1 = egocentric_view + (matplotlib.cm.jet(saliency)[:,:,:3]*255)
    img2 = img1 / np.max(img1)
    img2 *= 255
    del gradients
    del saliency
    del img1
    return img2

def write_over_gradcam_maps(img, text, frame_count, mode, wrong=False):
    gradcam_output_img = Image.fromarray(img.astype(np.uint8))
    d = ImageDraw.Draw(gradcam_output_img)
    if wrong:
        d.text((10,10), str(text), fill=(255,0,0))
        d.text((10, 200), str(frame_count), fill=(255,0,0))
    else:
        d.text((10,10), str(mode), fill=(0,0,0)) 
        d.text((10,30), str(text), fill=(0,0,0)) # Black Font
        d.text((10, 200), str(frame_count), fill=(0,0,0)) # Black Font
    gradcam_with_text = np.array(gradcam_output_img)
    return gradcam_with_text

def generate_video(
    video_option: List[str],
    video_dir: Optional[str],
    images: List[np.ndarray],
    episode_id: int,
    checkpoint_idx: int,
    metrics: Dict[str, float],
    tb_writer: TensorboardWriter,
    fps: int = 10,
) -> None:
    r"""Generate video according to specified information.

    Args:
        video_option: string list of "tensorboard" or "disk" or both.
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        episode_id: episode id for video naming.
        checkpoint_idx: checkpoint index for video naming.
        metric_name: name of the performance metric, e.g. "spl".
        metric_value: value of metric.
        tb_writer: tensorboard writer object for uploading video.
        fps: fps for generated video.
    Returns:
        None
    """
    if len(images) < 1:
        return

    metric_strs = []
    for k, v in metrics.items():
        metric_strs.append(f"{k}={v:.2f}")

    video_name = f"episode={episode_id}-ckpt={checkpoint_idx}-" + "-".join(
        metric_strs
    )
    if "disk" in video_option:
        assert video_dir is not None
        images_to_video(images, video_dir, video_name)
    if "tensorboard" in video_option:
        tb_writer.add_video_from_np_images(
            f"episode{episode_id}", checkpoint_idx, images, fps=fps
        )


def observations_to_image(observation: Dict, info: Dict) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    egocentric_view = []
    if "rgb" in observation:
        observation_size = observation["rgb"].shape[0]
        egocentric_view.append(observation["rgb"][:, :, :3])

    # draw depth map if observation has depth info
    if "depth" in observation:
        observation_size = observation["depth"].shape[0]
        depth_map = (observation["depth"].squeeze() * 255).astype(np.uint8)
        depth_map = np.stack([depth_map for _ in range(3)], axis=2)
        egocentric_view.append(depth_map)

    assert (
        len(egocentric_view) > 0
    ), "Expected at least one visual sensor enabled."
    egocentric_view = np.concatenate(egocentric_view, axis=1)

    # draw collision
    if "collisions" in info and info["collisions"]["is_collision"]:
        egocentric_view = draw_collision(egocentric_view)

    frame = egocentric_view

    if "top_down_map" in info:
        top_down_map = info["top_down_map"]["map"]
        top_down_map = maps.colorize_topdown_map(
            top_down_map, info["top_down_map"]["fog_of_war_mask"]
        )
        map_agent_pos = info["top_down_map"]["agent_map_coord"]
        top_down_map = maps.draw_agent(
            image=top_down_map,
            agent_center_coord=map_agent_pos,
            agent_rotation=info["top_down_map"]["agent_angle"],
            agent_radius_px=top_down_map.shape[0] // 16,
        )

        if top_down_map.shape[0] > top_down_map.shape[1]:
            top_down_map = np.rot90(top_down_map, 1)

        # scale top down map to align with rgb view
        old_h, old_w, _ = top_down_map.shape
        top_down_height = observation_size
        top_down_width = int(float(top_down_height) / old_h * old_w)
        # cv2 resize (dsize is width first)
        top_down_map = cv2.resize(
            top_down_map,
            (top_down_width, top_down_height),
            interpolation=cv2.INTER_CUBIC,
        )
        frame = np.concatenate((egocentric_view, top_down_map), axis=1)
    return frame, egocentric_view, top_down_map