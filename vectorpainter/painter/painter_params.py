# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

import numpy as np
import omegaconf
import pathlib
import pydiffvg
import random
import torch
from scipy.spatial import ConvexHull
from skimage.segmentation import mark_boundaries, slic
from vectorpainter.diffvg_warp import DiffVGState
from vectorpainter.token2attn.ptp_utils import view_images
from tqdm import trange


class Painter(DiffVGState):

    def __init__(
            self,
            cfg: omegaconf.DictConfig,
            diffvg_cfg: omegaconf.DictConfig,
            content_img,
            style_img,
            style_dir,
            num_strokes=4,
            num_segments=4,
            canvas_size=224,
            device=None,
    ):
        super(Painter, self).__init__(device, print_timing=diffvg_cfg.print_timing,
                                      canvas_width=canvas_size, canvas_height=canvas_size)
        self.content_img = content_img
        self.style_img = style_img
        self.style_dir = style_dir

        self.device = device

        self.num_paths = num_strokes
        self.num_segments = num_segments
        self.width = cfg.width
        self.max_width = cfg.max_width
        self.optim_width = cfg.optim_width
        self.control_points_per_seg = cfg.control_points_per_seg
        self.optim_rgba = cfg.optim_rgba
        self.optim_alpha = cfg.optim_opacity
        self.num_stages = cfg.num_stages

        self.shapes = []
        self.shape_groups = []
        self.num_control_points = 0
        self.color_vars_threshold = cfg.color_vars_threshold

        self.path_svg = cfg.path_svg
        self.strokes_per_stage = self.num_paths
        self.optimize_flag = []

        self.strokes_counter = 0  # counts the number of calls to "get_path"

    def init_image(self, random=False):
        if not random:
            # style_img (1, 3, M, N) -> (M, N, 3)
            style_img = self.style_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            # brush stroke init
            segments = self.segment_style(style_img, self.num_paths)
            s, e, c, width, color = self.clusters_to_strokes(
                segments,
                style_img,
                self.canvas_height,
                self.canvas_width,
            )

            self.num_control_points = torch.zeros(self.num_segments, dtype=torch.int32) \
                                      + (self.control_points_per_seg - 2)
            for i in range(s.shape[0]):
                points = []
                points.append((s[i][0], s[i][1]))
                points.append((c[i][0], c[i][1]))
                points.append((e[i][0], e[i][1]))

                points = torch.tensor(points).to(self.device)
                path = pydiffvg.Path(num_control_points=self.num_control_points,
                                     points=points,
                                     stroke_width=torch.tensor(width[i]),
                                     is_closed=False)
                self.shapes.append(path)
                self.strokes_counter += 1

                stroke_color = torch.tensor([color[i][0], color[i][1], color[i][2], 1.0])
                path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(self.shapes) - 1]),
                                                 fill_color=None,
                                                 stroke_color=stroke_color)
                self.shape_groups.append(path_group)
            self.optimize_flag = [True for _ in range(len(self.shapes))]

            for path in self.shapes:
                path.stroke_width.data.clamp_(1.0, self.max_width)

        else:
            num_paths_exists = 0
            if self.path_svg is not None and pathlib.Path(self.path_svg).exists():
                print(f"-> init svg from `{self.path_svg}` ...")

                self.canvas_width, self.canvas_height, self.shapes, self.shape_groups = self.load_svg(self.path_svg)
                # if you want to add more strokes to existing ones and optimize on all of them
                num_paths_exists = len(self.shapes)

            for i in range(num_paths_exists, self.num_paths):
                stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
                path = self.get_path()
                self.shapes.append(path)
                path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(self.shapes) - 1]),
                                                 fill_color=None,
                                                 stroke_color=stroke_color)
                self.shape_groups.append(path_group)
            self.optimize_flag = [True for i in range(len(self.shapes))]

        img = self.render_warp()
        img = img[:, :, 3:4] * img[:, :, :3] + \
              torch.ones(img.shape[0], img.shape[1], 3, device=self.device) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        img = img.unsqueeze(0)  # convert img from HWC to NCHW
        img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW
        return img

    def segment_style(self, style_img, num_paths):

        segments = slic(
            style_img,
            n_segments=num_paths,
            compactness=2,
            sigma=1,
            start_label=0
        )
        segmented_image = mark_boundaries(style_img, segments) * 255
        view_images(segmented_image, save_image=True, fp=self.style_dir / 'segmented.png')

        return segments

    def clusters_to_strokes(self, segments, img, H, W):
        segments += np.abs(np.min(segments))
        num_clusters = np.max(segments)
        clusters_params = {
            's': [],
            'e': [],
            'width': [],
            'color_rgb': []
        }

        print('start extracting stroke parameters...')

        for cluster_idx in trange(num_clusters + 1):
            cluster_mask = segments == cluster_idx
            if np.sum(cluster_mask) < 5:
                continue
            cluster_mask_nonzeros = np.nonzero(cluster_mask)

            cluster_points = np.stack((cluster_mask_nonzeros[0], cluster_mask_nonzeros[1]), axis=-1)
            try:
                convex_hull = ConvexHull(cluster_points)
            except:
                continue

            # find the two points (pixels) in the cluster that have the largest distance between them
            border_points = cluster_points[convex_hull.simplices.reshape(-1)]
            dist = np.sum((np.expand_dims(border_points, axis=1) - border_points) ** 2, axis=-1)
            max_idx_a, max_idx_b = np.nonzero(dist == np.max(dist))
            point_a = border_points[max_idx_a[0]]
            point_b = border_points[max_idx_b[0]]
            v_ab = point_b - point_a

            distances = np.zeros(len(border_points))
            for i, point in enumerate(border_points):
                v_ap = point - point_a
                distance = np.abs(np.cross(v_ab, v_ap)) / np.linalg.norm(v_ab)
                distances[i] = distance
            average_width = np.mean(distances)

            if average_width == 0.0:
                continue

            clusters_params['s'].append(point_a / img.shape[:2])
            clusters_params['e'].append(point_b / img.shape[:2])
            clusters_params['width'].append(average_width)

            clusters_params['color_rgb'].append(np.mean(img[cluster_mask], axis=0))

        for key in clusters_params.keys():
            clusters_params[key] = np.array(clusters_params[key])

        s = clusters_params['s']
        e = clusters_params['e']
        width = clusters_params['width']
        color = clusters_params['color_rgb']

        s[..., 0] *= H
        s[..., 1] *= W
        e[..., 0] *= H
        e[..., 1] *= W
        c = (s + e) / 2.

        s, e, c, width, color = [x.astype(np.float32) for x in [s, e, c, width, color]]
        s = s[..., ::-1]
        e = e[..., ::-1]
        c = c[..., ::-1]

        return s, e, c, width, color

    def get_path(self):
        self.num_control_points = torch.zeros(self.num_segments, dtype=torch.int32) + (self.control_points_per_seg - 2)
        points = []
        p0 = (random.random(), random.random())
        points.append(p0)

        for j in range(self.num_segments):
            radius = 0.05
            for k in range(self.control_points_per_seg - 1):
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                points.append(p1)
                p0 = p1
        points = torch.tensor(points).to(self.device)
        points[:, 0] *= self.canvas_width
        points[:, 1] *= self.canvas_height

        path = pydiffvg.Path(num_control_points=self.num_control_points,
                             points=points,
                             stroke_width=torch.tensor(self.width),
                             is_closed=False)
        self.strokes_counter += 1
        return path

    def get_image(self):
        img = self.render_warp()

        opacity = img[:, :, 3:4]
        img = opacity * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=self.device) * (1 - opacity)
        img = img[:, :, :3]
        img = img.unsqueeze(0)  # convert img from HWC to NCHW
        img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW
        return img

    def clip_curve_shape(self):
        if self.optim_width:
            for path in self.shapes:
                path.stroke_width.data.clamp_(1.0, self.max_width)
        if self.optim_rgba:
            for group in self.shape_groups:
                group.stroke_color.data.clamp_(0.0, 1.0)
                # group.stroke_color.data[-1].clamp_(1.0, 1.0)  # force full opacity
        else:
            if self.optim_alpha:
                for group in self.shape_groups:
                    # group.stroke_color.data: RGBA
                    group.stroke_color.data[:3].clamp_(0., 0.)  # to force black stroke
                    group.stroke_color.data[-1].clamp_(0., 1.)  # opacity

    def path_pruning(self):
        for group in self.shape_groups:
            group.stroke_color.data[-1] = (group.stroke_color.data[-1] >= self.color_vars_threshold).float()

    def set_points_parameters(self):
        # stoke`s location optimization
        self.point_vars = []
        for i, path in enumerate(self.shapes):
            if self.optimize_flag[i]:
                path.points.requires_grad = True
                self.point_vars.append(path.points)

    def get_points_params(self):
        return self.point_vars

    def set_width_parameters(self):
        # stroke`s  width optimization
        self.width_vars = []
        for i, path in enumerate(self.shapes):
            if self.optimize_flag[i]:
                path.stroke_width.requires_grad = True
                self.width_vars.append(path.stroke_width)

    def get_width_parameters(self):
        return self.width_vars

    def set_color_parameters(self):
        # for strokes color optimization (opacity)
        self.color_vars = []
        for i, group in enumerate(self.shape_groups):
            if self.optimize_flag[i]:
                group.stroke_color.requires_grad = True
                self.color_vars.append(group.stroke_color)

    def get_color_parameters(self):
        return self.color_vars

    def save_svg(self, output_dir, fname):
        pydiffvg.save_svg(f'{output_dir}/{fname}.svg',
                          self.canvas_width,
                          self.canvas_height,
                          self.shapes,
                          self.shape_groups)

    @staticmethod
    def softmax(x, tau=0.2):
        e_x = np.exp(x / tau)
        return e_x / e_x.sum()


class SketchPainterOptimizer:

    def __init__(
            self,
            renderer: Painter,
            points_lr: float,
            optim_alpha: bool,
            optim_rgba: bool,
            color_lr: float,
            optim_width: bool,
            width_lr: float
    ):
        self.renderer = renderer

        self.points_lr = points_lr
        self.optim_color = optim_alpha or optim_rgba
        self.color_lr = color_lr
        self.optim_width = optim_width
        self.width_lr = width_lr

        self.points_optimizer, self.width_optimizer, self.color_optimizer = None, None, None

    def init_optimizers(self):
        self.renderer.set_points_parameters()
        self.points_optimizer = torch.optim.Adam(self.renderer.get_points_params(), lr=self.points_lr)
        if self.optim_color:
            self.renderer.set_color_parameters()
            self.color_optimizer = torch.optim.Adam(self.renderer.get_color_parameters(), lr=self.color_lr)
        if self.optim_width:
            self.renderer.set_width_parameters()
            self.width_optimizer = torch.optim.Adam(self.renderer.get_width_parameters(), lr=self.width_lr)

    def update_lr(self, step, decay_steps=(500, 750)):
        if step % decay_steps[0] == 0 and step > 0:
            for param_group in self.points_optimizer.param_groups:
                param_group['lr'] = 0.4
        if step % decay_steps[1] == 0 and step > 0:
            for param_group in self.points_optimizer.param_groups:
                param_group['lr'] = 0.1

    def zero_grad_(self):
        self.points_optimizer.zero_grad()
        if self.optim_color:
            self.color_optimizer.zero_grad()
        if self.optim_width:
            self.width_optimizer.zero_grad()

    def step_(self):
        self.points_optimizer.step()
        if self.optim_color:
            self.color_optimizer.step()
        if self.optim_width:
            self.width_optimizer.step()

    def get_lr(self):
        return self.points_optimizer.param_groups[0]['lr']
