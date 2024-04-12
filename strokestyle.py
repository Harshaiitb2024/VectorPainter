# -*- coding: utf-8 -*-
# Author: ximing xing
# Description: the main func of this project.
# Copyright (c) 2023, XiMing Xing.

import os
import sys
from functools import partial

from accelerate.utils import set_seed
import hydra
import omegaconf

sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

from strokestyle.utils import render_batch_wrap, get_seed_range

METHODS = [
    'StrokeStyle',
    'StrokeStyleTransfer'
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
    if flag == "StrokeStyle":
        # read each line of prompts.txt file, use as prompt
        if cfg.prompt is None:
            with open("prompts.txt", 'r') as f:
                cfg.prompt = f.readlines()
                # shuffle
                # import random
                # random.shuffle(cfg.prompt)
        else:
            cfg.prompt = [cfg.prompt]

        from strokestyle.pipelines.StrokeStyle_pipeline import StrokeStylePipeline
        for prompt in cfg.prompt:
            cfg.prompt = prompt.strip()
            if not cfg.multirun:  # generate SVG multiple times
                pipe = StrokeStylePipeline(cfg)
                pipe.painterly_rendering(cfg.prompt, cfg.target)
            else:  # generate many SVG at once
                render_batch_fn(pipeline=StrokeStylePipeline, text_prompt=cfg.prompt, content_fpath=cfg.content, style_fpath=cfg.target)
    elif flag == "StylizedStrokeStyle":
        from StrokeStyle.strokestyle.pipelines.StrokeStyleTransfer_pipeline import StrokeStyleTransferPipeline
        if not cfg.multirun:
            pipe = StrokeStyleTransferPipeline(cfg)
            pipe.painterly_rendering(cfg.content, cfg.target)
        else:
            render_batch_fn(pipeline=StrokeStyleTransferPipeline, content_fpath=cfg.content, style_fpath=cfg.target)

if __name__ == '__main__':
    main()
