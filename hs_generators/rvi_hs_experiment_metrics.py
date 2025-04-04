from __future__ import annotations

from olimp.evaluation.loss.stress import STRESS
from olimp.evaluation.loss.corr import Correlation
from olimp.evaluation.loss.s_oklab import SOkLab
from olimp.evaluation.loss.flip import LDRFLIPLoss
from olimp.evaluation.loss.piq import MultiScaleSSIMLoss
from olimp.evaluation.loss.lpips import LPIPS
from olimp.evaluation.loss.rmse import RMSE


from olimp.precompensation.optimization.montalto import (
    MontaltoParameters,
    montalto,
)

from olimp.processing import resize_kernel, fft_conv
from .rvi_hs_experiment_methods import (
    select_random_img,
    select_random_psf,
    rescale_image,
)

from olimp.dataset.olimp import olimp
from olimp.dataset.sca_2023 import sca_2023

import torch
from torchvision.utils import save_image

from typing import Callable, Union, Dict

import argparse
import random
import json
from pathlib import Path

from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    BarColumn,
    TextColumn,
)
from dataclasses import dataclass
import numpy as np


def pixels_to_inches(
    img_px_w: int | None = None,
    diagonal_inch: float = 24.0,
    aspect_w: int = 16,
    aspect_h: int = 9,
    img_percent_of_screen: float = 0.26,
) -> float:
    """
    Переводит ширину объекта из пикселей или процентов в дюймы.

    :param obj_px_w: ширина объекта в пикселях
    :param diagonal_inch: диагональ экрана в дюймах
    :param aspect_w: ширина соотношения сторон экрана
    :param aspect_h: высота соотношения сторон экрана
    :param screen_width_px: ширина экрана в пикселях
    :param obj_percent: ширина объекта в процентах от ширины экрана
    :return: ширина объекта в дюймах
    """
    # Вычисляем ширину экрана в дюймах
    ratio = aspect_w / aspect_h
    screen_width_inch = (
        (diagonal_inch**2) * (ratio**2) / (1 + ratio**2)
    ) ** 0.5
    return img_px_w / (screen_width_inch * img_percent_of_screen)


def create_soklab_metric(target: torch.Tensor):
    img_dpi_w = pixels_to_inches(img_px_w=target.shape[-1])
    return SOkLab(dpi=img_dpi_w, distance_inch=19.685).to(device=target.device)


def select_random_metrics(metrics: tuple[MetricInfo]) -> list[str]:
    """Randomly selects two evaluation metric names."""
    return random.sample(metrics, 2)


def calc_metrics(
    img1: torch.Tensor,
    img2: torch.Tensor,
    metrics: tuple[MetricInfo, ...],
) -> dict[str, float]:
    results = {}
    for metric_info in metrics:
        value = metric_info.create_metric(img1)(
            img1.clip(0, 1), img2.clip(0, 1)
        )
        if isinstance(value, torch.Tensor):
            value = value.item()
        results[metric_info.name] = 1 - value

    return results


def load_olimp_dataset(category: str) -> dict[str, list[str]]:
    return olimp(categories={category})


def load_sca_dataset(category: str) -> Dict[str, list[str]]:
    return sca_2023(categories={category})


@dataclass
class MetricInfo:
    name: str
    create_metric: Callable[
        [torch.Tensor],
        Union[torch.nn.Module, Callable[..., Union[torch.Tensor, float]]],
    ]
    lr: float
    gap: float = 1e-4


metrics = (
    # This for debug
    # MetricInfo(
    #     name="RMSE_sRGB",
    #     create_metric=lambda _target: RMSE(color_space="srgb").to(
    #         device=_target.device
    #     ),
    #     lr=1e-1,
    #     gap=1e-2,
    # ),
    # MetricInfo(
    #     name="RMSE_CIELAB",
    #     create_metric=lambda _target: RMSE(color_space="lab").to(
    #         device=_target.device
    #     ),
    #     lr=1e-1,
    #     gap=1e-2,
    # ),
    # MetricInfo(
    #     name="RMSE_proLab",
    #     create_metric=lambda _target: RMSE(color_space="prolab").to(
    #         device=_target.device
    #     ),
    #     lr=1e-1,
    #     gap=1e-2,
    # ),
    # MetricInfo(
    #     name="RMSE_Oklab",
    #     create_metric=lambda _target: RMSE(color_space="oklab").to(
    #         device=_target.device
    #     ),
    #     lr=1e-1,
    #     gap=1e-2,
    # ),
    # real metrics
    MetricInfo(
        name="stress",
        create_metric=lambda _target: STRESS(invert=False).to(
            device=_target.device
        ),
        lr=1e-1,
    ),  # good
    MetricInfo(
        name="corr",
        create_metric=lambda _target: Correlation(invert=True).to(
            device=_target.device
        ),
        lr=1e-2,
    ),  # good
    MetricInfo(
        name="flip",
        create_metric=lambda _target: LDRFLIPLoss().to(device=_target.device),
        lr=1e-2,
        gap=1e-5,
    ),  # good
    MetricInfo(
        name="ms-ssim",
        create_metric=lambda _target: MultiScaleSSIMLoss(reduction="mean").to(
            device=_target.device
        ),
        lr=1e-3,
    ),  # good
    MetricInfo(
        name="lpips",
        create_metric=lambda _target: LPIPS(net="alex").to(
            device=_target.device
        ),
        lr=1e-2,
    ),
    MetricInfo(
        name="soklab", create_metric=create_soklab_metric, lr=1e-2, gap=1e-5
    ),  # function that immediately calculates the value
)


def main():
    parser = argparse.ArgumentParser(
        description="Run precompensation metrics evaluation."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of image-PSF pairs to process.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=52,
        help="Number of image-PSF pairs to process.",
    )
    args = parser.parse_args()

    random.seed(args.seed)  # 52 12 7
    torch.manual_seed(args.seed)  # 52 12 7

    output_dir = Path(f"hs_content/metrics/")
    output_dir.mkdir(parents=True, exist_ok=True)

    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    olimp_img_dataset: dict[str, list[str]] = load_olimp_dataset("*")
    sca_psf_dataset: dict[str, list[str]] = load_sca_dataset("PSFs/Broad")

    progress = Progress(
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TimeElapsedColumn(),
        TextColumn("{task.description}"),
    )
    Progress.get_default_columns()

    with progress, device:
        task_ds = progress.add_task("Processing...", total=args.n)

        for i in range(args.n):
            img, img_path = select_random_img(
                olimp_img_dataset, "*", device=device
            )
            psf, psf_path = select_random_psf(
                sca_psf_dataset, "PSFs/Broad", device
            )

            img = img[:, :3, ...]
            img_scaled = rescale_image(img)
            psf_scaled = resize_kernel(psf, img_scaled.shape[-2:])
            psf_scaled_shift = torch.fft.fftshift(psf_scaled)
            psf_scaled_shift /= torch.sum(psf_scaled_shift.flatten())

            pair_dir = output_dir / f"pair_{i}"
            pair_dir.mkdir(exist_ok=True, parents=True)

            psf_filename = pair_dir / "psf.npy"
            np.save(
                psf_filename.with_suffix(".npy"),
                psf_scaled.cpu().detach().numpy(),
            )

            img_filename = pair_dir / "target.png"
            save_image(img_scaled.cpu(), img_filename)

            img_filename = pair_dir / "blured.png"
            save_image(
                fft_conv(img_scaled, psf_scaled_shift).cpu(), img_filename
            )

            json_filename = pair_dir / "metrics.json"
            json_results = []

            for metric_info in select_random_metrics(metrics=metrics):
                print(f"Running metric: {metric_info.name}")
                torch.cuda.empty_cache()
                loss_func = metric_info.create_metric(img_scaled)
                if isinstance(loss_func, torch.nn.Module):
                    loss_func = loss_func.to(device)

                params = MontaltoParameters(
                    loss_func=loss_func, lr=metric_info.lr, gap=metric_info.gap
                )
                precomp = montalto(
                    image=img_scaled, psf=psf_scaled_shift, parameters=params
                )

                retinal_precomp_img = fft_conv(precomp, psf_scaled_shift)

                img_filename = pair_dir / f"{metric_info.name}.png"
                save_image(retinal_precomp_img.cpu(), img_filename)

                calc_result = calc_metrics(
                    img1=img_scaled,
                    img2=retinal_precomp_img,
                    metrics=metrics,
                )

                json_entry = {
                    "image_path": str(img_path),
                    "psf_path": str(psf_path),
                    "metric_used_for_precompensation": metric_info.name,
                    "calculated_metrics": calc_result,
                }
                json_results.append(json_entry)

            with open(json_filename, "w") as f:
                json.dump(json_results, f, indent=4)

            progress.advance(task_ds)


if __name__ == "__main__":
    main()
