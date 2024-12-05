# -*- coding: utf-8 -*-
# Author: ximing xing
# Description: the main func of this project.
# Copyright (c) 2023, XiMing Xing.

from functools import partial

from accelerate.utils import set_seed
import hydra
import omegaconf

from vectorpainter.utils import render_batch_wrap, get_seed_range

METHODS = [
    'VectorPainter',
]


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(cfg: omegaconf.DictConfig):
    """
    The project configuration is stored in './conf/config.yaml'
    And style configurations are stored in './conf/x/stroke.yaml'
    """
    flag = cfg.x.method
    assert flag in METHODS, f"{flag} is not currently supported!"

    # set seed
    set_seed(cfg.seed)
    seed_range = get_seed_range(cfg.srange) if cfg.multirun else None

    # render function
    render_batch_fn = partial(render_batch_wrap, cfg=cfg, seed_range=seed_range)
    if flag == "VectorPainter":
        # read each line of prompts.txt file, use as prompt
        if cfg.prompt is None:
            with open("prompts.txt", 'r') as f:
                cfg.prompt = f.readlines()
                # shuffle
                # import random
                # random.shuffle(cfg.prompt)
        else:
            cfg.prompt = [cfg.prompt]

        from vectorpainter.pipelines import VectorPainterPipeline

        for prompt in cfg.prompt:
            cfg.prompt = prompt.strip()
            if not cfg.multirun:  # generate SVG multiple times
                pipe = VectorPainterPipeline(cfg)
                pipe.painterly_rendering(text_prompt=cfg.prompt,
                                         negative_prompt=cfg.neg_prompt,
                                         style_fpath=cfg.style,
                                         style_prompt=cfg.style_prompt)
            else:  # generate many SVG at once
                render_batch_fn(pipeline=VectorPainterPipeline,
                                text_prompt=cfg.prompt,
                                negative_prompt=cfg.neg_prompt,
                                style_fpath=cfg.style,
                                style_prompt=cfg.style_prompt)


if __name__ == '__main__':
    main()
