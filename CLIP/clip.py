import json
import logging
import os
import pathlib
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
from .model import CLIP, CustomTextCLIP, convert_weights_to_lp, convert_to_custom_text_state_dict, resize_pos_embed, get_cast_dtype
from .openai import load_openai_model


_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs
_MODEL_CKPT_PATHS = {'ViT-L-14-336': Path(__file__).parent / "ckpt/ViT-L-14-336px.pt"}


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_rescan_model_configs()  # initial populate of model config registry


def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())



def get_model_config(model_name):
    # print(_MODEL_CONFIGS)
    if model_name in _MODEL_CONFIGS:
        # print('herehere')
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None


def load_state_dict(checkpoint_path: str, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict



def load_checkpoint(model, checkpoint_path, strict=True):
    state_dict = load_state_dict(checkpoint_path)
    # detect old format and make compatible with new format
    if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
        state_dict = convert_to_custom_text_state_dict(state_dict)
    resize_pos_embed(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


def create_model(
        model_name: str,
        img_size: int,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        output_dict: Optional[bool] = None,
        require_pretrained: bool = False,
        adapter = False,
):

    model_name = model_name.replace('/', '-')  # for callers using old naming with / in ViT names
    checkpoint_path = None
    model_cfg = None

    if isinstance(device, str):
        device = torch.device(device)

    if pretrained and pretrained.lower() == 'openai':
        logging.info(f'Loading pretrained {model_name} from OpenAI.')
        model_cfg = model_cfg or get_model_config(model_name)
        # print(model_cfg['vision_cfg'])
        if model_cfg['vision_cfg']['image_size'] != img_size:
            model_cfg['vision_cfg']['image_size'] = img_size
            cast_dtype = get_cast_dtype(precision)

            model_pre = load_openai_model(
                name = _MODEL_CKPT_PATHS[model_name],
                precision=precision,
                device=device,
                jit=jit,
            )
            state_dict = model_pre.state_dict()

            # to always output dict even if it is clip
            if output_dict and hasattr(model_pre, "output_dict"):
                model_pre.output_dict = True

            model = CLIP(**model_cfg, cast_dtype=cast_dtype)
            ### for resnet
            if not hasattr(model.visual, 'grid_size'):
                model.visual.grid_size = int(np.sqrt(model.visual.attnpool.positional_embedding.shape[0] - 1))
            resize_pos_embed(state_dict, model)
            incompatible_keys = model.load_state_dict(state_dict, strict=True)
            model.to(device=device)
            if precision in ("fp16", "bf16"):
                convert_weights_to_lp(model, dtype=torch.bfloat16 if precision == 'bf16' else torch.float16)

            # set image / mean metadata from pretrained_cfg if available, or use default
            model.visual.image_mean = (0.48145466, 0.4578275, 0.40821073)
            model.visual.image_std = (0.26862954, 0.26130258, 0.27577711)

            # to always output dict even if it is clip
            if output_dict and hasattr(model, "output_dict"):
                model.output_dict = True

            if jit:
                model = torch.jit.script(model)
        else:
            model = load_openai_model(
                model_name,
                precision=precision,
                device=device,
                jit=jit,
                cache_dir=cache_dir,
            )

            # to always output dict even if it is clip
            if output_dict and hasattr(model, "output_dict"):
                model.output_dict = True
    else:
        # print('here')
        model_cfg = model_cfg or get_model_config(model_name)
        if model_cfg is not None:
            print(f'Loaded {model_name} model config.')
        else:
            raise RuntimeError(f'Model config for {model_name} not found.')

        if force_quick_gelu:
            # override for use of QuickGELU on non-OpenAI transformer models
            model_cfg["quick_gelu"] = True

        if force_patch_dropout is not None:
            # override the default patch dropout value
            model_cfg["vision_cfg"]["patch_dropout"] = force_patch_dropout

        if force_image_size is not None:
            # override model config's image size
            model_cfg["vision_cfg"]["image_size"] = force_image_size


        cast_dtype = get_cast_dtype(precision)
        custom_text = model_cfg.pop('custom_text', False) or force_custom_text

        if custom_text:
            model = CustomTextCLIP(**model_cfg, cast_dtype=cast_dtype)
        else:
            model = CLIP(**model_cfg, cast_dtype=cast_dtype)

        pretrained_loaded = False
        if pretrained:
            checkpoint_path = _MODEL_CKPT_PATHS[model_name]
            if checkpoint_path:
                print(f'Loading pretrained {model_name} weights ({pretrained}).')
                load_checkpoint(model, checkpoint_path)
            else:
                raise RuntimeError(f'Pretrained weights ({pretrained}) not found for model {model_name}.')
            pretrained_loaded = True

        if require_pretrained and not pretrained_loaded:
            # callers of create_model_from_pretrained always expect pretrained weights
            raise RuntimeError(
                f'Pretrained weights were required for (model: {model_name}, pretrained: {pretrained}) but not loaded.')

        model.to(device=device)
        if precision in ("fp16", "bf16"):
            convert_weights_to_lp(model, dtype=torch.bfloat16 if precision == 'bf16' else torch.float16)

        # set image / mean metadata from pretrained_cfg if available, or use default
        model.visual.image_mean = (0.48145466, 0.4578275, 0.40821073)
        model.visual.image_std = (0.26862954, 0.26130258, 0.27577711)

        # to always output dict even if it is clip
        if output_dict and hasattr(model, "output_dict"):
            model.output_dict = True

        if jit:
            model = torch.jit.script(model)
    
    return model
