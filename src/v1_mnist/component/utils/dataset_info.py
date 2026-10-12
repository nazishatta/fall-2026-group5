"""What the SOM code needs to know about the dataset and the CNN, read from the config.

The class count, the embedding size and the names used in figure titles and
file names all come from the config (base.yaml: the dataset and model
sections; every SOM config merges base.yaml). The SOM code itself has no
MNIST values written into it, so the same code runs on another dataset by
changing the config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DatasetInfo:
    name: str                 # dataset.name, e.g. "mnist"
    display_name: str         # dataset.display_name, e.g. "MNIST" (figure titles)
    num_classes: int          # dataset.num_classes, e.g. 10
    class_label: str          # dataset.class_label, e.g. "digit" (titles and file names)
    model_display_name: str   # model.display_name, e.g. "LeNet" (figure titles)
    feature_layer: str        # model.feature_layer, e.g. "fc2"
    feature_dim: int          # model.feature_dim, e.g. 84


def _required(section: Any, section_name: str, key: str) -> Any:
    value = getattr(section, key, None)
    if value is None:
        raise KeyError(f"The config has no {section_name}.{key} (see configs/base.yaml).")
    return value


def dataset_info(config: Any) -> DatasetInfo:
    """Read the dataset and model facts from a loaded config."""
    dataset = getattr(config, "dataset", None)
    model = getattr(config, "model", None)
    if dataset is None or model is None:
        raise KeyError("The config has no dataset or model section (see configs/base.yaml).")

    name = str(_required(dataset, "dataset", "name"))
    return DatasetInfo(
        name=name,
        display_name=str(getattr(dataset, "display_name", None) or name.upper()),
        num_classes=int(_required(dataset, "dataset", "num_classes")),
        class_label=str(getattr(dataset, "class_label", None) or "class"),
        model_display_name=str(
            getattr(model, "display_name", None) or _required(model, "model", "architecture")
        ),
        feature_layer=str(_required(model, "model", "feature_layer")),
        feature_dim=int(_required(model, "model", "feature_dim")),
    )
