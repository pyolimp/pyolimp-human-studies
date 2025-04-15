from __future__ import annotations

from olimp.precompensation.basic.huang import huang
from olimp.precompensation.optimization.ji import ji, JiParameters
from olimp.precompensation.optimization.montalto import (
    montalto,
    MontaltoParameters,
)
from olimp.precompensation.nn.models.usrnet import PrecompensationUSRNet
from olimp.precompensation.nn.models.unet_efficient_b0 import (
    PrecompensationUNETB0,
)

from olimp.dataset.sca_2023 import sca_2023
from olimp.dataset import read_img_path, ProgressContext

from olimp.processing import resize_kernel, fft_conv, scale_value

import torch
from torchvision.transforms import transforms
from torchvision.utils import save_image

from typing import Any, Callable
from itertools import product

import argparse
import random
import json
from pathlib import Path
from contextlib import contextmanager

from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    BarColumn,
    TextColumn,
)

methods: dict[str, Callable[[Any, Any], Any]] = {
    "blur": lambda img, psf: fft_conv(img, psf),
    "huang": lambda img, psf: scale_value(huang(img, psf), 0, 1),
    "ji": lambda img, psf: ji(img, psf, JiParameters()),
    "montalto": lambda img, psf: montalto(img, psf, MontaltoParameters()),
    "usrnet": PrecompensationUSRNet.from_path(path="hf://RVI/usrnet.pth"),
    "unet_b0": PrecompensationUNETB0.from_path(
        "hf://RVI/unet-efficientnet-b0.pth"
    ),
}


def load_sca_dataset(
    category: str, progress_context: ProgressContext
) -> dict[str, list[str]]:
    return sca_2023(categories={category}, progress_context=progress_context)


def select_random_img(
    dataset: dict[str, list[str]], category: str, device: torch.device
) -> tuple[torch.Tensor, str]:
    get_length: int = len(dataset[category])
    img_idx: int = random.randint(0, get_length - 1)
    img: torch.Tensor = read_img_path(dataset[category][img_idx])
    return (
        img.unsqueeze(0).to(device=device) / 255.0,
        dataset[category][img_idx],
    )


def select_random_psf(
    dataset: dict[str, list[str]], category: str, device: torch.device
) -> tuple[torch.Tensor, str]:
    get_length: int = len(dataset[category])
    psf_idx: int = random.randint(0, get_length - 1)
    psf: torch.Tensor = read_img_path(dataset[category][psf_idx])
    return (
        psf.unsqueeze(0).to(device=device) / psf.sum(),
        dataset[category][psf_idx],
    )


def rescale_image(image, max_size=512):
    _, _, h, w = image.shape
    if max(h, w) <= max_size:
        return image

    if h > w:
        new_h = max_size
        new_w = int(w * (max_size / h))
    else:
        new_w = max_size
        new_h = int(h * (max_size / w))

    transform = transforms.Resize((new_h, new_w), antialias=True)
    return transform(image)


def main():
    parser = argparse.ArgumentParser(
        description="Data Generation for HS via RVI precompensation methods"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--n", type=int, default=5, help="Number of iterations per category"
    )
    parser.add_argument(
        "--img_categories",
        nargs="+",
        help="List of image categories to process. If provided, these will override the default ones.",
        default=[
            "Images/Icons",
            "Images/Real_images/Animals",
            "Images/Real_images/Faces",
            "Images/Real_images/Natural",
            "Images/Real_images/Urban",
            "Images/Texts",
        ],
    )
    parser.add_argument(
        "--psf_categories",
        nargs="+",
        help="List of PSF categories to process. If provided, these will override the default ones.",
        default=["PSFs/Broad", "PSFs/Medium", "PSFs/Narrow"],
    )

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    img_categories: list[str] = (
        args.img_categories if args.img_categories else img_categories
    )
    psf_categories: list[str] = (
        args.psf_categories if args.psf_categories else psf_categories
    )

    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    progress = Progress(
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TimeElapsedColumn(),
        TextColumn("{task.description}"),
    )
    Progress.get_default_columns()
    with device, progress:
        task = progress.add_task(
            "[cyan]Processing Images...",
            total=len(img_categories) * len(psf_categories),
        )
        task_ds = progress.add_task("Dataset...", total=1.0)

        @contextmanager
        def progress_callback():
            yield lambda description, completed: progress.update(
                task_ds, completed=completed, description=description
            )

        for img_cat, psf_cat in product(img_categories, psf_categories):
            progress.update(task_ds, completed=0.0, description="Loading...")

            olimp_img_dataset: dict[str, list[str]] = load_sca_dataset(
                img_cat, progress_callback()
            )
            sca_psf_dataset: dict[str, list[str]] = load_sca_dataset(
                psf_cat, progress_callback()
            )

            print(f"Select category: {img_cat} | {psf_cat}")

            img_cat_short = img_cat.split("/")[-1]
            psf_cat_short = psf_cat.split("/")[-1]
            output_dir = Path(
                f"hs_content/methods/{psf_cat_short}/{img_cat_short}"
            )

            output_dir.mkdir(parents=True, exist_ok=True)

            for i in range(args.n):
                img, img_path = select_random_img(
                    olimp_img_dataset, img_cat, device=device
                )
                psf, psf_path = select_random_psf(
                    sca_psf_dataset, psf_cat, device
                )

                img_gray = img.mean(dim=1, keepdim=True)
                img_scaled = rescale_image(img_gray)
                psf_scaled = resize_kernel(psf, img.shape[-2:])
                psf_scaled_shift = torch.fft.fftshift(psf_scaled)

                pair_dir = output_dir / f"pair_{i}"
                pair_dir.mkdir(parents=True, exist_ok=True)

                psf_filename = pair_dir / "psf.npy"
                torch.save(psf_scaled.cpu(), psf_filename)

                img_filename = pair_dir / "target.png"
                save_image(img_scaled.cpu(), img_filename)

                for method_name, method_func in methods.items():
                    with torch.device(device):
                        if isinstance(method_func, torch.nn.Module):
                            method_func.to(device)

                            if method_name == "usrnet":
                                preprocessed = method_func.preprocess(
                                    img_scaled.repeat(1, 3, 1, 1),
                                    psf_scaled_shift,
                                )
                            else:
                                preprocessed = method_func.preprocess(
                                    img_scaled, psf_scaled_shift
                                )
                            precomp = method_func(preprocessed)[0]
                            retinal_precomp = fft_conv(
                                precomp, psf_scaled_shift
                            )
                        else:
                            precomp = method_func(img_scaled, psf_scaled_shift)
                            retinal_precomp = fft_conv(
                                precomp, psf_scaled_shift
                            )

                    retinal_filename = pair_dir / f"{method_name}.png"
                    save_image(retinal_precomp.cpu(), retinal_filename)

                    data = {
                        f"{psf_cat_short}_{img_cat_short}": [
                            {
                                "img_path": str(img_path),
                                "psf_path": str(psf_path),
                            }
                        ]
                    }

                    (pair_dir / "paths.json").write_text(
                        json.dumps(data, indent=4)
                    )

            progress.update(task, advance=1)


if __name__ == "__main__":
    main()
