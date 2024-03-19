# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

import numpy as np
import omegaconf
import pydiffvg
import torch
from scipy.spatial import ConvexHull
from skimage.segmentation import mark_boundaries, slic
from strokestyle.diffvg_warp import DiffVGState


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

    def init_image(self, stage=0):
        # style_img (1, 3, M, N) -> (M, N, 3)
        style_img = self.style_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        # brush stroke init
        segments = self.segment_style(style_img, self.num_paths)
        location, s, e, c, width, color = self.clusters_to_strokes(
            segments,
            style_img,
            self.canvas_height,
            self.canvas_width,
            sec_scale=1.1,
            width_scale=0.1
        )
        s += location
        c += location
        e += location

        self.num_control_points = torch.zeros(self.num_segments, dtype=torch.int32) + (self.control_points_per_seg - 2)
        for i in range(location.shape[0]):
            points = []
            points.append((s[i][0], s[i][1]))
            points.append((c[i][0], c[i][1]))
            points.append((e[i][0], e[i][1]))

            points = torch.tensor(points).to(self.device)
            path = pydiffvg.Path(num_control_points=torch.tensor(self.num_control_points),
                                 points=points,
                                 stroke_width=torch.tensor(width[i][0]),
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
            min_size_factor=0.02,
            max_size_factor=4.,
            compactness=2,
            sigma=1,
            start_label=0
        )
        mark_boundaries(style_img, segments) * 255
        return segments

    def clusters_to_strokes(self, segments, img, H, W, sec_scale=0.001, width_scale=1):
        segments += np.abs(np.min(segments))
        num_clusters = np.max(segments)
        clusters_params = {
            'center': [],
            's': [],
            'e': [],
            'bp1': [],
            'bp2': [],
            'num_pixels': [],
            'stddev': [],
            'width': [],
            'color_rgb': []
        }

        for cluster_idx in range(num_clusters + 1):
            cluster_mask = segments == cluster_idx
            if np.sum(cluster_mask) < 5: continue
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
            # compute the two intersection points of the line that goes orthogonal to point_a and point_b
            v_ba = point_b - point_a
            v_orth = np.array([v_ba[1], -v_ba[0]])
            m = (point_a + point_b) / 2.0
            n = m + 0.5 * v_orth
            p = cluster_points[convex_hull.simplices][:, 0]
            q = cluster_points[convex_hull.simplices][:, 1]
            u = - ((m[..., 0] - n[..., 0]) * (m[..., 1] - p[..., 1]) - (m[..., 1] - n[..., 1]) * (
                    m[..., 0] - p[..., 0])) \
                / ((m[..., 0] - n[..., 0]) * (p[..., 1] - q[..., 1]) - (m[..., 1] - n[..., 1]) * (
                    p[..., 0] - q[..., 0]))
            intersec_idcs = np.logical_and(u >= 0, u <= 1)
            intersec_points = p + u.reshape(-1, 1) * (q - p)
            intersec_points = intersec_points[intersec_idcs]

            width = np.sum((intersec_points[0] - intersec_points[1]) ** 2)

            if width == 0.0: continue

            clusters_params['s'].append(point_a / img.shape[:2])
            clusters_params['e'].append(point_b / img.shape[:2])
            clusters_params['bp1'].append(intersec_points[0] / img.shape[:2])
            clusters_params['bp2'].append(intersec_points[1] / img.shape[:2])
            clusters_params['width'].append(np.sum((intersec_points[0] - intersec_points[1]) ** 2))

            clusters_params['color_rgb'].append(np.mean(img[cluster_mask], axis=0))
            center_x = np.mean(cluster_mask_nonzeros[0]) / img.shape[0]
            center_y = np.mean(cluster_mask_nonzeros[1]) / img.shape[1]
            clusters_params['center'].append(np.array([center_x, center_y]))
            clusters_params['num_pixels'].append(np.sum(cluster_mask))
            clusters_params['stddev'].append(np.mean(np.std(img[cluster_mask], axis=0)))

        for key in clusters_params.keys():
            clusters_params[key] = np.array(clusters_params[key])

        N = clusters_params['center'].shape[0]

        stddev = clusters_params['stddev']
        rel_num_pixels = 5 * clusters_params['num_pixels'] / np.sqrt(H * W)

        location = clusters_params['center']
        num_pixels_per_cluster = clusters_params['num_pixels'].reshape(-1, 1)
        s = clusters_params['s']
        e = clusters_params['e']
        cluster_width = clusters_params['width']

        location[..., 0] *= H
        location[..., 1] *= W
        s[..., 0] *= H
        s[..., 1] *= W
        e[..., 0] *= H
        e[..., 1] *= W

        s -= location
        e -= location

        color = clusters_params['color_rgb']

        c = (s + e) / 2. + np.stack([np.random.uniform(low=-1, high=1, size=[N]),
                                     np.random.uniform(low=-1, high=1, size=[N])],
                                    axis=-1)

        sec_center = (s + e + c) / 3.
        s -= sec_center
        e -= sec_center
        c -= sec_center

        rel_num_pix_quant = np.quantile(rel_num_pixels, q=[0.3, 0.99])
        width_quant = np.quantile(cluster_width, q=[0.3, 0.99])
        rel_num_pixels = np.clip(rel_num_pixels, rel_num_pix_quant[0], rel_num_pix_quant[1])
        cluster_width = np.clip(cluster_width, width_quant[0], width_quant[1])
        width = width_scale * rel_num_pixels.reshape(-1, 1) * cluster_width.reshape(-1, 1)
        s, e, c = [x * sec_scale for x in [s, e, c]]

        location, s, e, c, width, color = [x.astype(np.float32) for x in [location, s, e, c, width, color]]
        location = location[..., ::-1]
        s = s[..., ::-1]
        e = e[..., ::-1]
        c = c[..., ::-1]

        return location, s, e, c, width, color

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
                group.stroke_color.data[-1].clamp_(1.0, 1.0)  # force full opacity
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
