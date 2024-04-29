import shutil
import time
from functools import partial
from pathlib import Path
from typing import AnyStr

import diffusers
import numpy as np
import torch
import torch.nn.functional as F
from moviepy.editor import VideoFileClip
from PIL import Image
from vectorpainter.diffusers_warp import init_StableDiffusion_pipeline, model2res
from vectorpainter.libs.engine import ModelState
from vectorpainter.libs.metric.clip_score import CLIPScoreWrapper
from vectorpainter.libs.metric.lpips_origin import LPIPS
from vectorpainter.painter import (LSDSPipeline, LSDSSDXLPipeline, Painter,
                                   SketchPainterOptimizer, get_relative_pos)
from vectorpainter.painter.sketch_utils import fix_image_scale
from vectorpainter.utils.plot import plot_couple, plot_img
from torchvision import transforms
from tqdm.auto import tqdm


class VectorPainterTransferPipeline(ModelState):
    def __init__(self, args):
        logdir_ = f"seed{args.seed}-im{args.x.image_size}" \
                  f"-{args.x.model_id}" \
                  f"-{time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))}"
        super().__init__(args, log_path_suffix=logdir_)

        self.result_path.mkdir(parents=True, exist_ok=True)
        self.print(f"results path: {self.result_path}")

        # create log dir
        self.style_dir = self.result_path / "style_image"
        self.sd_sample_dir = self.result_path / "sd_sample"
        self.png_logs_dir = self.result_path / "png_logs"
        self.svg_logs_dir = self.result_path / "svg_logs"
        self.log_dir = self.result_path / "configs"

        # use dir name to save logs
        self.clip_vis_loss_dir = self.log_dir / f"clip_vis_loss_({args.x.clip.vis_loss})"
        self.style_coeff_dir = self.log_dir / f"style_coeff_({args.x.perceptual.style_coeff})"
        self.pos_loss_weight_dir = self.log_dir / f"pos_loss_weight_({args.x.pos_loss_weight})"

        if self.accelerator.is_main_process:
            self.style_dir.mkdir(parents=True, exist_ok=True)
            self.sd_sample_dir.mkdir(parents=True, exist_ok=True)
            self.png_logs_dir.mkdir(parents=True, exist_ok=True)
            self.svg_logs_dir.mkdir(parents=True, exist_ok=True)

            self.clip_vis_loss_dir.mkdir(parents=True, exist_ok=True)
            self.style_coeff_dir.mkdir(parents=True, exist_ok=True)
            self.pos_loss_weight_dir.mkdir(parents=True, exist_ok=True)

        self.make_video = self.args.mv
        if self.make_video:
            self.frame_idx = 0
            self.frame_log_dir = self.result_path / "frame_logs"
            self.frame_log_dir.mkdir(parents=True, exist_ok=True)

        if self.x_cfg.model_id == "sdxl":
            custom_pipeline = LSDSSDXLPipeline
            # custom_scheduler = diffusers.DPMSolverMultistepScheduler
            custom_scheduler = diffusers.EulerDiscreteScheduler
        else:  # sd21, sd14, sd15
            custom_pipeline = LSDSPipeline
            custom_scheduler = diffusers.DDIMScheduler

        self.diffusion = init_StableDiffusion_pipeline(
            self.x_cfg.model_id,
            custom_pipeline=custom_pipeline,
            custom_scheduler=custom_scheduler,
            device=self.device,
            local_files_only=not args.diffuser.download,
            force_download=args.diffuser.force_download,
            resume_download=args.diffuser.resume_download,
            ldm_speed_up=self.x_cfg.ldm_speed_up,
            enable_xformers=self.x_cfg.enable_xformers,
            gradient_checkpoint=self.x_cfg.gradient_checkpoint,
        )

        self.g_device = torch.Generator(device=self.device).manual_seed(args.seed)

        # init clip model and clip score wrapper
        self.cargs = self.x_cfg.clip
        self.clip_score_fn = CLIPScoreWrapper(
            self.cargs.model_name,
            device=self.device,
            visual_score=True,
            feats_loss_type=self.cargs.feats_loss_type,
            feats_loss_weights=self.cargs.feats_loss_weights,
            fc_loss_weight=self.cargs.fc_loss_weight
        )

    def painterly_rendering(self, content_fpath, style_fpath):
        content_img = Image.open(content_fpath).convert("RGB")
        process_comp = transforms.Compose([
            transforms.Resize(size=(self.x_cfg.image_size, self.x_cfg.image_size))
        ])
        content_img = process_comp(content_img)

        # process style file
        style_img = self.load_and_process_style_file(style_fpath)
        plot_img(style_img, self.style_dir, fname="style_image_preprocess")

        perceptual_loss_fn = None
        if self.x_cfg.perceptual.content_coeff > 0 or self.x_cfg.perceptual.style_coeff > 0:
            lpips_loss_fn = LPIPS(net=self.x_cfg.perceptual.lpips_net).to(self.device)
            perceptual_loss_fn = partial(lpips_loss_fn.forward, return_per_layer=False, normalize=False)

        inputs = self.get_target(
            content_img,
            self.x_cfg.image_size,
            self.x_cfg.fix_scale
        )
        inputs = inputs.detach()  # inputs as GT
        self.print("inputs shape: ", inputs.shape)

        # load renderer
        renderer = self.load_render(inputs, style_img)
        img = renderer.init_image(random=self.x_cfg.random_init)
        self.print("init_image shape: ", img.shape)
        plot_img(img, self.style_dir, fname="init_style")
        renderer.save_svg(self.style_dir.as_posix(), "init_style")

        # load optimizer
        optimizer = SketchPainterOptimizer(renderer,
                                           self.x_cfg.lr,
                                           self.x_cfg.optim_opacity,
                                           self.x_cfg.optim_rgba,
                                           self.x_cfg.color_lr,
                                           self.x_cfg.optim_width,
                                           self.x_cfg.width_lr)
        optimizer.init_optimizers()

        # log params
        init_relative_pos = get_relative_pos(renderer.get_points_params()).detach()  # init stroke position as GT

        self.print(f"-> Painter points Params: {len(renderer.get_points_params())}")
        self.print(f"-> Painter width Params: {len(renderer.get_width_parameters())}")
        self.print(f"-> Painter color Params: {len(renderer.get_color_parameters())}")

        total_iter = self.x_cfg.num_iter
        self.print(f"\ntotal optimization steps: {total_iter}")
        with tqdm(initial=self.step, total=total_iter, disable=not self.accelerator.is_main_process) as pbar:
            while self.step < total_iter:
                raster_sketch = renderer.get_image().to(self.device)

                if self.make_video and (self.step % self.args.framefreq == 0 or self.step == total_iter - 1):
                    plot_img(raster_sketch, self.frame_log_dir, fname=f"iter{self.frame_idx}")
                    self.frame_idx += 1

                # CLIP data augmentation
                raster_sketch_aug, inputs_aug = self.clip_pair_augment(
                    raster_sketch, inputs,
                    im_res=224,
                    augments=self.cargs.augmentations,
                    num_aug=self.cargs.num_aug
                )

                # clip visual loss
                total_visual_loss = torch.tensor(0.)
                l_clip_fc, l_clip_conv, clip_conv_loss_sum = torch.tensor(0), [], torch.tensor(0)
                if self.x_cfg.clip.vis_loss > 0:
                    l_clip_fc, l_clip_conv = self.clip_score_fn.compute_visual_distance(
                        raster_sketch_aug, inputs_aug, clip_norm=False
                    )
                    clip_conv_loss_sum = sum(l_clip_conv)
                    total_visual_loss = self.x_cfg.clip.vis_loss * (clip_conv_loss_sum + l_clip_fc)

                # perceptual loss with style image
                l_percep_style = torch.tensor(0.)
                if self.x_cfg.perceptual.style_coeff > 0 and perceptual_loss_fn is not None:
                    l_perceptual = perceptual_loss_fn(style_img, raster_sketch).mean()
                    l_percep_style = l_perceptual * self.x_cfg.perceptual.style_coeff

                # prep with inputs
                l_percep_content = torch.tensor(0.)
                if self.x_cfg.perceptual.content_coeff > 0 and perceptual_loss_fn is not None:
                    l_perceptual_ = perceptual_loss_fn(inputs, raster_sketch).mean()
                    l_percep_content = l_perceptual_ * self.x_cfg.perceptual.content_coeff

                # control points relative position loss
                l_rel_pos = torch.tensor(0.)
                if self.x_cfg.pos_loss_weight > 0:
                    l_rel_pos = F.mse_loss(
                        get_relative_pos(renderer.get_points_params()), init_relative_pos
                    ).mean() * self.x_cfg.pos_loss_weight

                # total loss
                loss = total_visual_loss + l_percep_style + l_percep_content + l_rel_pos

                # optimization
                optimizer.zero_grad_()
                loss.backward()
                optimizer.step_()

                # update lr
                if self.x_cfg.lr_scheduler:
                    optimizer.update_lr(self.step, self.x_cfg.decay_steps)

                # records
                pbar.set_description(
                    f"lr: {optimizer.get_lr():.2f}, "
                    f"l_total: {loss.item():.4f}"
                )

                # log raster and svg
                if self.step % self.args.save_step == 0 and self.accelerator.is_main_process:
                    # log png
                    plot_couple(inputs,
                                raster_sketch,
                                self.step,
                                output_dir=self.png_logs_dir.as_posix(),
                                fname=f"iter{self.step}")
                    # log svg
                    renderer.save_svg(self.svg_logs_dir.as_posix(), f"svg_iter{self.step}")

                self.step += 1
                pbar.update(1)

        # saving final result
        renderer.save_svg(self.result_path.as_posix(), "final_svg")

        final_raster_sketch = renderer.get_image().to(self.device)
        plot_img(final_raster_sketch, self.result_path, fname='final_render')

        if self.make_video:
            video_path = self.result_path / "rendering.mp4"
            from subprocess import call
            call([
                "ffmpeg",
                "-framerate", f"{self.args.framerate}",
                "-i", (self.frame_log_dir / "iter%d.png").as_posix(),
                "-vb", "20M",
                video_path.as_posix()
            ])
            video = VideoFileClip(video_path.as_posix())
            video.write_gif((self.result_path / "rendering.gif").as_posix(), fps=video.fps)

        self.close(msg="painterly rendering complete.")

    def load_and_process_style_file(self, style_fpath):
        style_path = Path(style_fpath)
        assert style_path.exists(), f"{style_fpath} is not exist!"
        # load style file
        style_img = self.style_file_preprocess(style_path.as_posix())
        self.print(f"load style file from: {style_path.as_posix()}")
        shutil.copy(style_path, self.style_dir)  # copy style file
        return style_img

    def style_file_preprocess(self, style_path):
        process_comp = transforms.Compose([
            transforms.Resize(size=(self.x_cfg.image_size, self.x_cfg.image_size)),
            transforms.ToTensor(),
            # transforms.Lambda(lambda t: t - 0.5),
            transforms.Lambda(lambda t: t.unsqueeze(0)),
            # transforms.Lambda(lambda t: (t + 1) / 2),
        ])

        style_pil = Image.open(style_path).convert("RGB")  # open file
        style_file = process_comp(style_pil)  # preprocess
        style_file = style_file.to(self.device)
        return style_file

    def get_target(self,
                   target,
                   image_size,
                   fix_scale):

        if target.mode == "RGBA":
            # Create a white rgba background
            new_image = Image.new("RGBA", target.size, "WHITE")
            # Paste the image on the background.
            new_image.paste(target, (0, 0), target)
            target = new_image
        target = target.convert("RGB")

        if fix_scale:
            target = fix_image_scale(target)

        # define image transforms
        transforms_ = []
        if target.size[0] != target.size[1]:
            transforms_.append(transforms.Resize((image_size, image_size)))
        else:
            transforms_.append(transforms.Resize(image_size))
            transforms_.append(transforms.CenterCrop(image_size))
        transforms_.append(transforms.ToTensor())

        # preprocess
        data_transforms = transforms.Compose(transforms_)
        target_ = data_transforms(target).unsqueeze(0).to(self.device)

        return target_

    def load_render(self, content_img, style_img):
        renderer = Painter(
            self.x_cfg,
            self.args.diffvg,
            content_img=content_img,
            style_img=style_img,
            style_dir=self.style_dir,
            num_strokes=self.x_cfg.num_paths,
            num_segments=self.x_cfg.num_segments,
            canvas_size=self.x_cfg.image_size,
            device=self.device
        )
        return renderer

    @property
    def clip_norm_(self):
        return transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    def clip_pair_augment(self,
                          x: torch.Tensor,
                          y: torch.Tensor,
                          im_res: int,
                          augments: str = "affine_norm",
                          num_aug: int = 4):
        # init augmentations
        augment_list = []
        if "affine" in augments:
            augment_list.append(
                transforms.RandomPerspective(fill=0, p=1.0, distortion_scale=0.5)
            )
            augment_list.append(
                transforms.RandomResizedCrop(im_res, scale=(0.8, 0.8), ratio=(1.0, 1.0))
            )
        augment_list.append(self.clip_norm_)  # CLIP Normalize

        # compose augmentations
        augment_compose = transforms.Compose(augment_list)
        # make augmentation pairs
        x_augs, y_augs = [self.clip_score_fn.normalize(x)], [self.clip_score_fn.normalize(y)]
        # repeat N times
        for n in range(num_aug):
            augmented_pair = augment_compose(torch.cat([x, y]))
            x_augs.append(augmented_pair[0].unsqueeze(0))
            y_augs.append(augmented_pair[1].unsqueeze(0))
        xs = torch.cat(x_augs, dim=0)
        ys = torch.cat(y_augs, dim=0)
        return xs, ys
