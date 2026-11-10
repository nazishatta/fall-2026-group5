"""Controlled autoencoder / Isolation Forest check on SVHN, as a brand-new run.

Varies training size and input scaling one at a time, adds a length-only control
score, and compares against the earlier unified run. The heavy step (CNN
embeddings of SVHN, the autoencoder) needs PyTorch and the SVHN files, so it is
imported inside ``load_embeddings`` and ``get_autoencoder_fitter``. Everything
else is plain numpy and is covered by unit tests. It never touches earlier
outputs and refuses to reuse a run ID.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.autoencoder_scaling import (
    AutoencoderFitter,
    check_design,
    evaluate_conditions,
    length_scores,
    summarize_conditions,
)
from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.unified_novelty_metrics import (
    evaluate,
    stratified_bootstrap,
)
from src.v1_mnist.component.utils.logging import get_logger
from src.v1_mnist.component.utils.repo_paths import resolve_inside
from src.v1_mnist.component.utils.som_io import sha256_file

ROOT = Path(__file__).resolve().parents[5]
CONFIG_PATH = ROOT / "src/v1_mnist/component/configs/autoencoder_scaling_diagnostic.yaml"
LOGGER = get_logger(__name__)


def repo_path(value: str) -> Path:
    """Resolve only safe, repository-relative paths."""
    return resolve_inside(ROOT, value)


def read_config(config_path: Path) -> dict[str, Any]:
    """Load the config and reject unsafe run IDs or an invalid design."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_id = config.get("run_id")
    if not isinstance(run_id, str) or not run_id or not all(ch.isalnum() or ch in "-_" for ch in run_id):
        raise ValueError("Unsafe run ID")
    check_design(
        [int(item) for item in config["training_sizes"]], list(config["scalings"]),
        [int(item) for item in config["seeds"]], int(config["expected"]["training_samples"]),
    )
    return config


def load_embeddings(config: dict[str, Any]) -> dict[str, Any]:
    """Frozen MNIST embeddings plus SVHN embeddings (needs torch, nnsom and the SVHN files)."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import SVHN

    from src.v1_mnist.component.p1_model_training_baseline.cnn_baseline_training.extractor import (
        FeatureExtractor,
    )
    from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.unified_novelty_benchmark import (
        load_inputs,
    )
    from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.ood_adapters import (
        image_transform,
    )
    from src.v1_mnist.component.p3_novelty_detection.qe_novelty_detection.run_qe_ood import (
        read_settings,
    )
    from src.v1_mnist.component.utils.shared_provenance import load_manifest

    train, validation, _labels, model, _som, _qe, provenance, hashes = load_inputs(config)
    svhn_path = repo_path(load_manifest()["svhn_file"])
    settings = read_settings(repo_path(config["qe_config"]))
    dataset = SVHN(
        root=str(svhn_path.parent), split="test", download=False,
        transform=image_transform("svhn", tuple(settings["image_size"])),
    )
    loader = DataLoader(dataset, batch_size=int(config["scoring"]["batch_size"]), shuffle=False)
    torch.set_num_threads(1)
    svhn, _, _ = FeatureExtractor(model, "fc2").extract(loader, torch.device("cpu"))
    return {"train": train, "validation": validation, "svhn": svhn, "hashes": hashes, "provenance": provenance}


def get_autoencoder_fitter() -> AutoencoderFitter:
    """The same autoencoder as the earlier benchmark runs (imported lazily: needs torch)."""
    from src.v1_mnist.component.p3_novelty_detection.novelty_benchmark.autoencoder_baseline import (
        fit_autoencoder,
    )

    return fit_autoencoder


def length_control(validation: np.ndarray, svhn: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    """AUROC of embedding length alone (higher length = more novel, like every other score)."""
    ids, oods = length_scores(validation), length_scores(svhn)
    target = float(config["target_tpr"])
    point = evaluate(ids, oods, target)
    interval = stratified_bootstrap(
        ids, oods, target, int(config["scoring"]["bootstrap_iterations"]), int(config["scoring"]["bootstrap_seed"])
    )
    return {
        **point,
        "auroc_interval_95": interval["auroc"],
        "mean_length_mnist_validation": float(ids.mean()),
        "mean_length_svhn": float(oods.mean()),
        "interpretation": (
            "Score = embedding length, higher = more novel. An AUROC far below 0.5 means SVHN "
            "embeddings are shorter than MNIST ones; 1 minus the AUROC then says how well "
            "'shorter = more novel' would separate them. This is a control for comparison, "
            "not a proposed detector."
        ),
    }


def compare_to_reference(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any] | None:
    """Largest per-seed AUROC gap to the earlier unified run, where the cells match."""
    path = repo_path(config["reference_metrics"])
    if not path.is_file():
        return None
    reference = json.loads(path.read_text(encoding="utf-8"))
    budget = int(reference["fitting"]["common_training_rows_per_seed"])
    result: dict[str, Any] = {"reference_training_rows": budget, "reference_file": config["reference_metrics"]}
    for method, scaling in (("autoencoder", "standardized"), ("isolation_forest", "standardized")):
        mine = {r["seed"]: r["auroc"] for r in rows if r["method"] == method and r["training_size"] == budget and r["scaling"] == scaling}
        theirs = {r["seed"]: r["metrics"][method]["auroc"] for r in reference["fitted_methods_per_seed"]}
        shared = sorted(set(mine) & set(theirs))
        result[method] = {
            "seeds_compared": shared,
            "max_abs_auroc_difference": max((abs(mine[s] - theirs[s]) for s in shared), default=None),
        }
    return result


def write_run(
    output: Path, config: dict[str, Any], config_sha256: str, data: dict[str, Any],
    rows: list[dict[str, Any]], control: dict[str, Any], reference_check: dict[str, Any] | None,
) -> dict[str, Any]:
    """Write every output of one run into a brand-new folder."""
    summary = summarize_conditions(rows)
    output.mkdir(parents=True, exist_ok=False)
    (output / "tables").mkdir()
    with (output / "tables" / "autoencoder_scaling_by_condition.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "training_size", "scaling", "auroc_mean", "auroc_sample_sd"])
        writer.writerows([[s["method"], s["training_size"], s["scaling"], s["auroc_mean"], s["auroc_sample_sd"]] for s in summary])
    result = {
        "run_id": config["run_id"],
        "study_type": "post_hoc_exploratory",
        "design": {
            "training_sizes": config["training_sizes"], "scalings": config["scalings"], "seeds": config["seeds"],
            "autoencoder": config["autoencoder"], "isolation_forest": config["isolation_forest"],
            "note": "Network and forest settings are copied from the earlier benchmark configs, not tuned.",
        },
        "condition_summary": summary,
        "per_seed_results": rows,
        "length_only_control": control,
        "reference_check": reference_check,
        "provenance": {"config_sha256": config_sha256, "input_sha256": data["hashes"], "frozen_som": data["provenance"]},
        "leakage_control": (
            "Everything is fitted on MNIST training rows only; scalers are fitted on the selected "
            "training rows only. MNIST validation and SVHN test are used for evaluation only. SVHN "
            "was examined in earlier runs, so this is exploratory. The final MNIST test split is "
            "never opened."
        ),
        "limitation": "Describes how the scores respond to size and scaling; it does not prove why SVHN embeddings are shorter.",
    }
    with (output / "autoencoder_scaling_metrics.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def run(config_path: Path) -> dict[str, Any]:
    """Run the full diagnostic under a new run ID; refuse to reuse an existing one."""
    config = read_config(config_path)
    output = repo_path(config["output_root"]) / config["run_id"]
    if output.exists():
        raise FileExistsError(f"STOP: Existing run: {output}")
    data = load_embeddings(config)
    expected = config["expected"]
    width = int(expected["embedding_dimensions"])
    if data["svhn"].shape != (int(expected["svhn_samples"]), width):
        raise RuntimeError("SVHN embedding shape mismatch")
    joined = np.concatenate((data["validation"], data["svhn"]), axis=0)
    rows = evaluate_conditions(data["train"], joined, len(data["validation"]), config, get_autoencoder_fitter())
    control = length_control(data["validation"], data["svhn"], config)
    result = write_run(
        output, config, sha256_file(config_path), data, rows, control, compare_to_reference(rows, config)
    )
    for cell in result["condition_summary"]:
        LOGGER.info("%s size=%d scaling=%s AUROC %.4f +/- %.4f", cell["method"], cell["training_size"],
                    cell["scaling"], cell["auroc_mean"], cell["auroc_sample_sd"])
    return result


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
