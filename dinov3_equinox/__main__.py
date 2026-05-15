import typing
from typing import Literal

import sys
from argparse import ArgumentParser
from pathlib import Path

import torch

# from dinov3.hub.backbones import (
#     dinov3_vit7b16,
#     dinov3_vitb16,
#     dinov3_vith16plus,
#     dinov3_vitl16,
#     dinov3_vits16,
#     dinov3_vits16plus,
# )
from safetensors.torch import save_file

Model = Literal["vit7b16", "vitb16", "vith16plus", "vitl16", "vits16", "vits16plus"]
Head = Literal["imagenet1k"]


def fix_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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


def convert_model(model_name: Model, weight_folder: Path, dest_folder: str, mmap: bool):
    dest_folder_ = Path(dest_folder)

    if not dest_folder_.exists():
        print(f"dest_folder {dest_folder} does not exist")
        return

    # model_out_file = f"{dest_folder}/{model}.safetensors"
    model_out_file = dest_folder_ / f"{model_name}.safetensors"

    if Path(model_out_file).exists():
        print(f"model file for model {model_name} ({model_out_file}) already exists, skipping")
        print("if you wish to re-convert the model, delete the file first")
        return

    print(f"loading model {model_name}")

    state_dict: dict[str, torch.Tensor]

    print(f"{mmap=}")

    match model_name:
        case "vit7b16":
            # dino = torch.hub.load(
            #     "facebookresearch/dinov3",
            #     "dinov3_vit7b16",
            #     weights=str(weight_folder / "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"),
            # )

            state_dict = torch.load(
                str(weight_folder / "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"), mmap=mmap
            )
        case "vitb16":
            # dino = torch.hub.load(
            #     "facebookresearch/dinov3",
            #     "dinov3_vitb16",
            #     weights=str(weight_folder / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
            # )

            state_dict = torch.load(
                str(weight_folder / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"), mmap=mmap
            )
        case "vith16plus":
            # dino = torch.hub.load(
            #     "facebookresearch/dinov3",
            #     "dinov3_vith16plus",
            #     weights=str(weight_folder / "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"),
            # )

            state_dict = torch.load(
                str(weight_folder / "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"), mmap=mmap
            )
        case "vitl16":
            # dino = torch.hub.load(
            #     "facebookresearch/dinov3",
            #     "dinov3_vitl16",
            #     weights=str(weight_folder / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
            # )

            state_dict = torch.load(
                str(weight_folder / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"), mmap=mmap
            )
        # case "vitl16plus":
        #     dino = dinov3_vitl16plus(
        #         pretrained=True,
        #         weights="???",
        #     )
        case "vits16":
            # dino = torch.hub.load(
            #     "facebookresearch/dinov3",
            #     "dinov3_vits16",
            #     weights=str(weight_folder / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
            # )

            state_dict = torch.load(
                str(weight_folder / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"), mmap=mmap
            )
        case "vits16plus":
            # dino = torch.hub.load(
            #     "facebookresearch/dinov3",
            #     "dinov3_vits16plus",
            #     weights=str(weight_folder / "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"),
            # )

            state_dict = torch.load(
                str(weight_folder / "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"), mmap=mmap
            )
        case _:
            raise ValueError(f"invalid model {model_name}")

    # assert isinstance(dino, torch.nn.Module)

    return

    print("fixing state dict")

    state_dict = fix_state_dict(state_dict)

    print("saving converted model")

    save_file(state_dict, model_out_file)

    print(f"saved converted model {model_name} in {model_out_file}")


def convert_head(head_name: Head, weight_dir: Path, dest_dir: str, mmap: bool):
    dest_folder_ = Path(dest_dir)

    if not dest_folder_.exists():
        print(f"dest_folder {dest_dir} does not exist")
        return

    # model_out_file = f"{dest_folder}/{model}.safetensors"
    model_out_file = dest_folder_ / f"{head_name}.safetensors"

    if Path(model_out_file).exists():
        print(f"model file for model {head_name} ({model_out_file}) already exists, skipping")
        print("if you wish to re-convert the model, delete the file first")
        return

    print(f"loading head {head_name}")

    print(f"{mmap=}")

    match head_name:
        case "imagenet1k":
            state_dict = torch.load(
                weight_dir / "dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth", mmap=mmap
            )
        case _:
            raise ValueError(f"invalid head {head_name}")

    print("fixing state dict")

    state_dict = fix_state_dict(state_dict)

    print("saving converted model")

    save_file(state_dict, model_out_file)

    print(f"saved converted model {head_name} in {model_out_file}")


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--model", type=str, choices=list(typing.get_args(Model)) + ["all"], required=False
    )
    parser.add_argument(
        "--head", type=str, choices=list(typing.get_args(Head)) + ["all"], required=False
    )

    parser.add_argument("--weight-dir", type=str, default="/media/LinuxData/models/dinov3")
    parser.add_argument("--dest-dir", type=str, default="./models.")

    parser.add_argument("--mmap", type=bool, default=True)

    args = parser.parse_args()

    model: Model | Literal["all"] = args.model
    head: Head | Literal["all"] = args.head

    weight_dir: str = args.weight_dir
    dest_dir: str = args.dest_dir

    mmap: bool = args.mmap

    del args

    if model is None and head is None or model is not None and head is not None:
        print("need to specify either --model or --head")
        sys.exit(1)

    weight_folder = Path(weight_dir)

    if not weight_folder.exists():
        print(f"weight folder {weight_folder} does not exist")
        sys.exit(1)

    if model is not None:
        if model == "all":
            for model_ in typing.get_args(Model):
                convert_model(
                    model_,
                    weight_folder=weight_folder,
                    dest_folder=dest_dir,
                    mmap=mmap,
                )
        else:
            assert model in typing.get_args(Model)

            convert_model(model, weight_folder=weight_folder, dest_folder=dest_dir, mmap=mmap)

    if head is not None:
        if head == "all":
            for head_ in typing.get_args(Head):
                convert_head(head_, weight_dir=weight_folder, dest_dir=dest_dir, mmap=mmap)
        else:
            assert head in typing.get_args(Head)

            convert_head(head, weight_dir=weight_folder, dest_dir=dest_dir, mmap=mmap)


if __name__ == "__main__":
    main()
