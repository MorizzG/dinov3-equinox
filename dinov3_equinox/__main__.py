import typing
from typing import Literal

import sys
from argparse import ArgumentParser
from pathlib import Path

from dinov3.hub.backbones import (
    dinov3_vit7b16,
    dinov3_vitb16,
    dinov3_vith16plus,
    dinov3_vitl16,
    dinov3_vits16,
    dinov3_vits16plus,
)
from safetensors.torch import save_file

Model = Literal["vit7b16", "vitb16", "vith16plus", "vitl16", "vitl16plus", "vits16", "vits16plus"]


def fix_state_dict(state_dict: dict) -> dict:
    fixed_state_dict = {}

    for key, value in state_dict.items():
        if key == "patch_embed.proj.bias":
            # equinox wants bias as (c, 1, 1)
            value = value[:, None, None]
        elif key == "cls_token" or key == "storage_tokens":
            # cls_token/storage_tokens in torch version has batch dim
            value = value[0, ...]

        assert key not in fixed_state_dict
        fixed_state_dict[key] = value

    return fixed_state_dict


def convert_model(model: Model, weight_folder: Path):
    model_out_file = f"models/{model}.safetensors"

    if Path(model_out_file).exists():
        print(f"model file for model {model} ({model_out_file}) already exists, skipping")
        print("if you wish to re-convert the model, delete the file first")
        return

    print(f"loading model {model}")

    match model:
        case "vit7b16":
            dino = dinov3_vit7b16(
                pretrained=True,
                weights=str(weight_folder / "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"),
            )
        case "vitb16":
            dino = dinov3_vitb16(
                pretrained=True,
                weights=str(weight_folder / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
            )
        case "vith16plus":
            dino = dinov3_vith16plus(
                pretrained=True,
                weights=str(weight_folder / "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"),
            )
        case "vitl16":
            dino = dinov3_vitl16(
                pretrained=True,
                weights=str(weight_folder / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
            )
        # case "vitl16plus":
        #     dino = dinov3_vitl16plus(
        #         pretrained=True,
        #         weights="???",
        #     )
        case "vits16":
            dino = dinov3_vits16(
                pretrained=True,
                weights=str(weight_folder / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
            )
        case "vits16plus":
            dino = dinov3_vits16plus(
                pretrained=True,
                weights=str(weight_folder / "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"),
            )
        case _:
            raise NotImplementedError

    print("fixing state dict")

    state_dict = fix_state_dict(dino.state_dict())

    print("saving converted model")

    save_file(state_dict, model_out_file)

    print(f"saved converted model {model} in {model_out_file}")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--model", type=str, choices=list(typing.get_args(Model)) + ["all"])

    parser.add_argument("--weight-folder", type=str, default="/media/LinuxData/models/dinov3")

    args = parser.parse_args()

    weight_folder = Path(args.weight_folder)

    if not weight_folder.exists():
        print(f"weight folder {weight_folder} does not exist")
        sys.exit(1)

    if args.model == "all":
        for model in typing.get_args(Model):
            convert_model(model, weight_folder)
    else:
        model: Model = args.model

        assert model in typing.get_args(Model)

        convert_model(model, weight_folder)
