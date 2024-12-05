import copy
import shutil
import time
from pathlib import Path

import diffusers
from diffusers import StableDiffusionXLPipeline
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from vectorpainter.diffusers_warp import init_StableDiffusion_pipeline
from vectorpainter.libs.engine import ModelState
from vectorpainter.painter import Painter, SketchPainterOptimizer, inversion, SinkhornLoss, \
    get_relative_pos, bezier_curve_loss
from vectorpainter.painter.sketch_utils import fix_image_scale
from vectorpainter.utils.plot import plot_couple, plot_img
from vectorpainter.utils import mkdirs, create_video

Tensor = torch.Tensor


class VectorPainterPipeline(ModelState):
    def __init__(self, args):
        logdir_ = f"seed{args.seed}-im{args.x.image_size}" \
                  f"-{args.x.model_id}" \
                  f"-{time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))}"
        super().__init__(args, log_path_suffix=logdir_)

        self.imit_cfg = self.x_cfg.imit_stage
        self.synt_cfg = self.x_cfg.synth_stage

        # create log dir
        self.style_dir = self.result_path / "style_image"
        self.sd_sample_dir = self.result_path / "sd_sample"
        self.imit_png_logs_dir = self.result_path / "imit_png_logs"
        self.imit_svg_logs_dir = self.result_path / "imit_svg_logs"
        self.png_logs_dir = self.result_path / "png_logs"
        self.svg_logs_dir = self.result_path / "svg_logs"

        mkdirs([self.result_path, self.style_dir, self.sd_sample_dir,
                self.imit_png_logs_dir, self.imit_svg_logs_dir,
                self.png_logs_dir, self.svg_logs_dir])
        self.print(f"results path: {self.result_path}")

        self.make_video = self.args.mv
        if self.make_video:
            self.frame_idx = 0
            self.frame_log_dir = self.result_path / "frame_logs"
            mkdirs([self.frame_log_dir])

        self.g_device = torch.Generator(device=self.device).manual_seed(args.seed)

    def painterly_rendering(self, text_prompt, negative_prompt, style_fpath, style_prompt):
        self.print(f"text prompt: {text_prompt}")
        self.print(f"negative prompt: {negative_prompt}")

        style_tensor = self.load_and_process_style_img(style_fpath)
        plot_img(style_tensor, self.style_dir, fname="style_image_input")
        self.print(f"style_input shape: {style_tensor.shape}")

        # load and init renderer
        renderer = self.load_render(style_tensor)
        img = renderer.init_canvas(random=self.x_cfg.random_init)
        plot_img(img, self.style_dir, fname="stroke_init_style")
        renderer.save_svg(self.style_dir.as_posix(), fname="stroke_init_style")

        # stage 1
        renderer, recon_style_fpath = self.brushstroke_imitation(renderer)
        # stage 2
        self.synthesis_with_style_supervision(text_prompt, negative_prompt, renderer, style_prompt, recon_style_fpath)

        # save the painting process as a video
        if self.make_video:
            self.print("\n making video...")
            video_path = self.result_path / "rendering.mp4"
            create_video(video_path, (self.frame_log_dir / "iter%d.png").as_posix(), self.args.framerate)

        self.close(msg="painterly rendering complete.")

    def brushstroke_imitation(self, renderer: Painter):
        # load optimizer
        optimizer = SketchPainterOptimizer(renderer,
                                           self.imit_cfg.lr,
                                           self.imit_cfg.color_lr,
                                           self.imit_cfg.width_lr,
                                           self.x_cfg.optim_opacity,
                                           self.x_cfg.optim_rgba,
                                           self.x_cfg.optim_width)
        optimizer.init_optimizers()

        self.print(f"-> Stoke Imitation Stage ...")
        self.print(f"-> Painter point params: {len(renderer.get_points_params())}")
        self.print(f"-> Painter width params: {len(renderer.get_width_parameters())}")
        self.print(f"-> Painter color params: {len(renderer.get_color_parameters())}")

        total_iter = self.imit_cfg.num_iter
        self.print(f"total optimization steps: {total_iter}")
        with tqdm(initial=self.step, total=total_iter, disable=not self.accelerator.is_main_process) as pbar:
            while self.step < total_iter:
                raster_sketch = renderer.get_image()
                loss = F.mse_loss(renderer.style_img, raster_sketch)

                # optimization
                optimizer.zero_grad_()
                loss.backward()
                optimizer.step_()

                # update lr
                if self.imit_cfg.lr_scheduler:
                    optimizer.update_lr(self.step, self.imit_cfg.decay_steps)

                # records
                pbar.set_description(
                    f"lr: {optimizer.get_lr():.2f}, "
                    f"l_total: {loss.item():.4f}"
                )

                # log raster and svg
                if self.step % self.args.save_step == 0 and self.accelerator.is_main_process:
                    plot_couple(renderer.style_img,
                                raster_sketch,
                                self.step,
                                output_dir=self.imit_png_logs_dir.as_posix(),
                                fname=f"iter{self.step}")
                    renderer.save_svg(self.imit_svg_logs_dir.as_posix(), fname=f"svg_iter{self.step}")

                self.step += 1
                pbar.update(1)

        # save style result
        renderer.save_svg(self.result_path.as_posix(), "style_result")
        style_result = renderer.get_image()
        recon_style_fpath = self.result_path / 'style_result.png'
        plot_img(style_result, self.result_path, fname='style_result')

        return renderer, recon_style_fpath

    def style_inversion(self, prompt, negative_prompt, style_prompt, init_fpath):
        init_img = Image.open(init_fpath).convert("RGB").resize((1024, 1024))

        # load pretrained diffusion model
        ldm_pipe = init_StableDiffusion_pipeline(
            self.x_cfg.model_id,
            custom_pipeline=StableDiffusionXLPipeline,
            custom_scheduler=diffusers.DDIMScheduler,
            device=self.device,
            torch_dtype=torch.float16,
            local_files_only=not self.args.diffuser.download,
            force_download=self.args.diffuser.force_download,
            ldm_speed_up=self.x_cfg.ldm_speed_up,
            enable_xformers=self.x_cfg.enable_xformers,
            gradient_checkpoint=self.x_cfg.gradient_checkpoint,
        )

        # ddim inversion
        x0 = copy.deepcopy(init_img)
        guidance_scale = 2
        zts = inversion.ddim_inversion(ldm_pipe, x0, style_prompt, self.x_cfg.num_inference_steps, guidance_scale)
        # zts: [ddim_steps+1, 4, w, h]
        zT, inversion_callback = inversion.make_inversion_callback(zts, offset=5)
        zT = zT.unsqueeze(0)

        # zT = zT / ldm_pipe.vae.config.scaling_factor
        # decode_zT = ldm_pipe.vae.decode(zT, return_dict=False)[0]
        # decode_zT = (decode_zT.cpu().detach() / 2 + 0.5).clamp(0, 1)
        # plot_img(decode_zT.float(), self.sd_sample_dir, fname='decode_zT')

        # instant style
        scale = {
            "up": {"block_0": [0.0, 1.0, 0.0]},
        }
        ldm_pipe.load_ip_adapter("h94/IP-Adapter",
                                 subfolder="sdxl_models",
                                 weight_name="ip-adapter_sdxl.bin",
                                 local_files_only=not self.args.diffuser.download)
        ldm_pipe.set_ip_adapter_scale(scale)

        prompts = [style_prompt, prompt, prompt]
        latents = torch.randn(len(prompts), 4, 128, 128,
                              device=self.device,
                              generator=self.g_device,
                              dtype=ldm_pipe.unet.dtype)
        latents[0] = zT
        latents[1] = zT.clone()

        outputs = ldm_pipe(
            prompt=prompts,
            negative_prompt=negative_prompt,
            height=1024,
            width=1024,
            ip_adapter_image=init_img,
            num_inference_steps=self.x_cfg.num_inference_steps,
            guidance_scale=self.x_cfg.guidance_scale,
            generator=self.g_device,
            latents=latents,
            add_watermarker=False,
            callback_on_step_end=inversion_callback,
            output_type='pt',
            return_dict=False
        )[0]
        self.print(outputs.shape)

        gen_file = self.sd_sample_dir / 'samples.png'
        # view_images([np.array(outputs)], save_image=True, fp=gen_file)
        save_image(outputs, fp=gen_file)
        target_file = self.sd_sample_dir / 'target.png'
        # view_images([np.array(outputs[-1])], save_image=True, fp=target_file)
        plot_img(outputs[-1], self.sd_sample_dir, fname='target')

        del ldm_pipe
        torch.cuda.empty_cache()

        target = Image.open(target_file)
        return target

    def synthesis_with_style_supervision(self, prompt, negative_prompt, renderer: Painter,
                                         style_prompt, recon_style_fpath: Path):
        # inversion
        target = self.style_inversion(prompt, negative_prompt, style_prompt, recon_style_fpath)
        inputs = self.get_target(target, self.x_cfg.image_size, self.x_cfg.fix_scale)
        inputs = inputs.detach()  # inputs as GT

        # log params
        init_relative_pos = get_relative_pos(renderer.get_points_params()).detach()  # init stroke position as GT
        init_curves = copy.deepcopy(renderer.get_points_params())

        # init optimizer
        optimizer = SketchPainterOptimizer(renderer,
                                           self.synt_cfg.lr,
                                           self.synt_cfg.color_lr,
                                           self.synt_cfg.width_lr,
                                           self.x_cfg.optim_opacity,
                                           self.x_cfg.optim_rgba,
                                           self.x_cfg.optim_width)
        optimizer.init_optimizers()

        self.print(f"\n-> Synthesis with Style Supervision ...")
        self.print(f"-> Painter points Params: {len(renderer.get_points_params())}")
        self.print(f"-> Painter width Params: {len(renderer.get_width_parameters())}")
        self.print(f"-> Painter color Params: {len(renderer.get_color_parameters())}")

        # init structure loss
        if self.x_cfg.struct_loss_weight > 0:
            if self.x_cfg.struct_loss == 'ssim':
                from vectorpainter.painter import SSIM
                l_struct_fn = SSIM()
            elif self.x_cfg.struct_loss == 'msssim':
                from vectorpainter.painter import MSSSIM
                l_struct_fn = MSSSIM()
            else:
                l_struct_fn = lambda x, y: torch.tensor(0.)  # zero loss
        # init shape loss
        sinkhorn_loss_fn = SinkhornLoss(device=self.device)

        self.step = 0
        total_iter = self.synt_cfg.num_iter
        self.print(f"total optimization steps: {total_iter}")
        with tqdm(initial=self.step, total=total_iter, disable=not self.accelerator.is_main_process) as pbar:
            while self.step < total_iter:
                raster_sketch = renderer.get_image()

                if self.make_video and (self.step % self.args.framefreq == 0 or self.step == total_iter - 1):
                    plot_img(raster_sketch, self.frame_log_dir, fname=f"iter{self.frame_idx}")
                    self.frame_idx += 1

                l_recon = torch.tensor(0.)
                if self.x_cfg.l2_loss > 0:
                    l_recon = F.mse_loss(raster_sketch, inputs) * self.x_cfg.l2_loss

                # Struct Loss
                l_struct = torch.tensor(0.)
                if self.x_cfg.struct_loss_weight > 0:
                    if self.x_cfg.struct_loss in ['ssim', 'msssim']:
                        l_struct = 1. - l_struct_fn(raster_sketch, inputs)
                    else:
                        l_struct = l_struct_fn(raster_sketch, inputs)

                # # prep with inputs
                # l_percep_content = torch.tensor(0.)
                # if self.x_cfg.perceptual.content_coeff > 0:
                #     l_perceptual_ = perceptual_loss_fn(inputs, raster_sketch).mean()
                #     l_percep_content = l_perceptual_ * self.x_cfg.perceptual.content_coeff

                # control points relative position loss
                l_rel_pos = torch.tensor(0.)
                if self.x_cfg.pos_loss_weight > 0:
                    if self.x_cfg.pos_type == 'pos':
                        l_rel_pos = F.mse_loss(get_relative_pos(renderer.get_points_params()),
                                               init_relative_pos) * self.x_cfg.pos_loss_weight
                    elif self.x_cfg.pos_type == 'bez':
                        l_rel_pos = bezier_curve_loss(renderer.get_points_params(),
                                                      init_curves) * self.x_cfg.pos_loss_weight
                    elif self.x_cfg.pos_type == 'sinkhorn':
                        l_rel_pos = sinkhorn_loss_fn(raster_sketch, renderer.style_img) * self.x_cfg.pos_loss_weight

                # total loss
                loss = l_recon + l_struct + l_rel_pos

                # optimization
                optimizer.zero_grad_()
                loss.backward()
                optimizer.step_()

                # update lr
                if self.synt_cfg.lr_scheduler:
                    optimizer.update_lr(self.step, self.synt_cfg.decay_steps)

                # records
                pbar.set_description(
                    f"lr: {optimizer.get_lr():.2f}, "
                    f"l_total: {loss.item():.4f}, "
                    f"l_recon: {l_recon.item():.4f}, "
                    f"l_struct: {l_struct.item():.4f}, "
                    f"l_pos: {l_rel_pos.item():.4f}"
                )

                # log raster and svg
                if self.step % self.args.save_step == 0 and self.accelerator.is_main_process:
                    # log png
                    plot_couple(inputs,
                                raster_sketch,
                                self.step,
                                output_dir=self.png_logs_dir.as_posix(),
                                fname=f"iter{self.step}",
                                prompt=prompt)
                    # log svg
                    renderer.save_svg(self.svg_logs_dir.as_posix(), fname=f"svg_iter{self.step}")

                self.step += 1
                pbar.update(1)

        # saving final result
        renderer.save_svg(self.result_path.as_posix(), "final_svg")
        final_raster_sketch = renderer.get_image().to(self.device)
        plot_img(final_raster_sketch, self.result_path, fname='final_render')

    def load_and_process_style_img(self, style_fpath):
        style_path = Path(style_fpath)
        assert style_path.exists(), f"{style_fpath} is not exist!"

        def _to_tensor(style_path):
            process_comp = transforms.Compose([
                transforms.Resize(size=(self.x_cfg.image_size, self.x_cfg.image_size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.unsqueeze(0)),
            ])

            style_pil = Image.open(style_path).convert("RGB")  # open file
            style_tensor = process_comp(style_pil).to(self.device)  # preprocess
            return style_tensor

        style_img = _to_tensor(style_path.as_posix())
        self.print(f"load style file from: {style_path.as_posix()}")
        shutil.copy(style_path, self.style_dir)  # copy style file
        return style_img

    def get_target(self, target, image_size, fix_scale):
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

    def load_render(self, style_img):
        renderer = Painter(
            self.x_cfg,
            self.args.diffvg,
            style_img=style_img,
            style_dir=self.style_dir,
            num_strokes=self.x_cfg.num_paths,
            num_segments=self.x_cfg.num_segments,
            canvas_size=self.x_cfg.image_size,
            device=self.device
        )
        return renderer
