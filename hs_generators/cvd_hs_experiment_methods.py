from __future__ import annotations

from olimp.precompensation.nn.models.cvd_swin.cvd_swin_1channel import (
    CVDSwin1Channel,
)
from olimp.precompensation.nn.models.cvd_swin.cvd_swin_3channels import (
    CVDSwin3Channels,
)
from olimp.precompensation.optimization.tennenholtz_zachevsky import (
    tennenholtz_zachevsky,
)
from olimp.precompensation.optimization.cvd_direct_optimization import (
    cvd_direct_optimization,
    CVDParameters,
)

from olimp.simulate.color_blindness_distortion import ColorBlindnessDistortion
from olimp.evaluation.loss.rms import RMS
from olimp.evaluation.loss.chromaticity_difference import (
    ChromaticityDifference,
)

from olimp.dataset.cvd import cvd as cvd_dataset
from olimp.dataset import read_img_path, ProgressContext


import torch
from torchvision.transforms import transforms
from torchvision.utils import save_image

from typing import Any, Callable

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
    "1ch_swin": CVDSwin1Channel.from_path(),
    "3ch_swin": CVDSwin3Channels.from_path(),
    "tennenholtz_zachevsky": lambda image, distortion: tennenholtz_zachevsky(
        image[0], distortion
    )[None],
    "cvd_direct_optimization": lambda image, distortion: cvd_direct_optimization(
        image, distortion, CVDParameters(loss_func=RMS("lab"))
    ),
}


def load_cvd_dataset(
    categiry: str, progress_context: ProgressContext
) -> dict[str, list[str]]:
    return cvd_dataset(
        categories={categiry}, progress_context=progress_context
    )


def select_random_image(
    dataset: dict[str, list[str]], category: str, device: torch.device
) -> tuple[torch.Tensor, str]:
    dataset_len = len(dataset[category])
    image_idx = random.randint(0, dataset_len - 1)
    image = read_img_path(dataset[category][image_idx])[:3]
    return (
        image.unsqueeze(0).to(device=device) / 255.0,
        dataset[category][image_idx],
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
        description="Data Generation for HS via CVD precompensation methods"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--n", type=int, default=5, help="Number of iterations per category"
    )
    parser.add_argument(
        "--img_categories",
        nargs="+",
        help="List of image categories to process. If provided, these will override the default ones.",
        default=["Color_cvd_D_experiment_100000"],
    )
    parser.add_argument(
        "--cvd_categories",
        nargs="+",
        help="List of CVD categories to process. If provided, there will override the default ones.",
        default=["protan", "deutan", "tritan"],
    )

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    image_categories = args.img_categories
    cvd_categories = args.cvd_categories

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            total=len(image_categories) * len(cvd_categories),
        )
        task_ds = progress.add_task("Dataset...", total=1.0)

        @contextmanager
        def progress_callback():
            yield lambda description, completed: progress.update(
                task_ds, completed=completed, description=description
            )

        RMS_lab = RMS("lab")
        RMS_prolab = RMS("prolab")
        CD_lab = ChromaticityDifference("lab")
        CD_prolab = ChromaticityDifference("prolab")

        for image_cat in image_categories:
            progress.update(task_ds, completed=0.0, description="Loading...")

            olimp_image_dataset: dict[str, list[str]] = load_cvd_dataset(
                image_cat, progress_callback()
            )

            for cvd_cat in cvd_categories:
                print(f"Select category: {image_cat} | {cvd_cat}")
                distortion = ColorBlindnessDistortion.from_type(cvd_cat)

                output_dir = Path(
                    f"hs_content/cvd_methods/{cvd_cat}/{image_cat}"
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                data = {}

                for i in range(args.n):
                    image, image_path = select_random_image(
                        olimp_image_dataset, image_cat, device=device
                    )
                    image_scaled = rescale_image(image)
                    image_sim = distortion()(image_scaled)

                    pair_dir = output_dir / f"pair_{i}"
                    pair_dir.mkdir(parents=True, exist_ok=True)

                    image_filename = pair_dir / "target.png"
                    save_image(image_scaled.cpu(), image_filename)
                    save_image(image_sim, pair_dir / "targer_sim.png")

                    for method_name, method_func in methods.items():
                        with torch.device(device):
                            if isinstance(method_func, torch.nn.Module):
                                method_func.to(device)
                                preprocessed = method_func.preprocess(
                                    image_scaled
                                )
                                precomp = method_func(preprocessed)
                                precomp = method_func.postprocess(precomp)[0]

                            else:
                                precomp = method_func(image_scaled, distortion)

                        precomp_sim = distortion()(precomp)
                        precomp_filename = pair_dir / f"{method_name}.png"
                        save_image(precomp.cpu(), precomp_filename)
                        save_image(
                            precomp_sim.cpu(),
                            pair_dir / f"{method_name}_sim.png",
                        )
                        data[f"{method_name}"] = {
                            "original": {
                                "CD_lab": CD_lab(image_scaled, precomp)
                                .cpu()
                                .item(),
                                "CD_prolab": CD_prolab(image_scaled, precomp)
                                .cpu()
                                .item(),
                            },
                            "simulated": {
                                "CD_lab": CD_lab(image_sim, precomp_sim)
                                .cpu()
                                .item(),
                                "CD_prolab": CD_prolab(image_sim, precomp_sim)
                                .cpu()
                                .item(),
                                "RMS_lab": RMS_lab(image_sim, precomp_sim)
                                .cpu()
                                .item(),
                                "RMS_prolab": RMS_prolab(
                                    image_sim, precomp_sim
                                )
                                .cpu()
                                .item(),
                            },
                        }
                    (pair_dir / "metrics.json").write_text(
                        json.dumps(data, indent=4)
                    )

                progress.update(task, advance=1)


if __name__ == "__main__":
    main()
