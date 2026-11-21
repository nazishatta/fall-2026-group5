"""The steps of the study that ``run_all.py`` runs after the core Phase 2 chain, in proposal order.

  Phase 2  error_pattern_baselines   SOM error cells against k-means and the other baselines
           error_detector_budget     the same-budget comparison of the error detectors
           som_optuna                Optuna search per grid, 5-seed confirmation, summary table
           feature_layer_ablation    fc1 against fc2: embeddings, 2 x 5 SOMs, comparison
  Phase 3  (novelty_detection: the three Phase 3 runners, planned in run_all.py)
           held_out_class_novelty    one network and map per held-out split, then the comparison
  Phase 4  guided_retraining         pools, transform check, both experiments trained and summarised
           guided_retraining_test    the single test pass, breakdown, SOM re-analysis, novelty after retraining

Every step reads its configs from the dataset's config folder (``--config-dir``); a step whose
config is not in that folder is not part of that dataset's study and is skipped with a message.
A step whose results all exist already is skipped too, as in Phase 3 (each step's run folders are
found by its own settings loader, so the run-ID rules live in one place). A half-finished step is
planned, and its script stops at the first part that exists, as in Phases 1 and 2.

Stop points the runner never passes on its own:
  - ``guided_retraining`` and ``guided_retraining_test`` are separate steps, and nothing runs the
    test step unless ``--to guided_retraining_test`` is asked for.
  - Between them a person fills in the macro-F1 guard (protocol and retraining.yaml) and freezes the
    protocol with ``run_protocol_stamp.py --config ...``. The runner does neither; the test stage
    itself refuses to open the test split until the stamp exists and both files still match it.
Each script keeps its own rule that a finished run is never overwritten.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.v1_mnist.component.pipeline.common import PIPELINE, REPO_ROOT, Step, resolve

COMPONENT = REPO_ROOT / "src" / "v1_mnist" / "component"
P2 = PIPELINE / "p2_som_training_visualization"
P3 = PIPELINE / "p3_novelty_detection"
P4 = PIPELINE / "p4_guided_retraining"
OPTUNA = COMPONENT / "p2_som_training_visualization" / "som_nnsom_training"


@dataclass(frozen=True)
class StudyStep:
    """One named step of ``run_all.py``: the configs it needs and the commands it runs."""

    name: str
    configs: tuple[str, ...]
    commands: Callable[[str], list[Step]]
    results: Callable[[str], list[Path]]

    def missing(self, config_dir: str) -> list[Path]:
        """Configs of this step that the dataset's folder does not have."""
        return [resolve(f"{config_dir}/{name}") for name in self.configs if not resolve(f"{config_dir}/{name}").is_file()]

    def finished(self, config_dir: str) -> list[Path] | None:
        """The step's results if every one of them exists, else None.

        A loader that cannot run (for example, inputs not pulled from DVC yet) means "not known to be
        finished": the step is planned and its own script reports the problem.
        """
        try:
            paths = self.results(config_dir)
        except Exception:  # noqa: BLE001 - any loader failure only means "plan the step"
            return None
        return paths if paths and all(path.exists() for path in paths) else None


def _staged(label: str, script: Path, config: str, stages: tuple[str, ...]) -> list[Step]:
    return [Step(f"{label}: {stage}", script, ["--config", config, "--stage", stage]) for stage in stages]


def _error_pattern_baselines(d: str) -> list[Step]:
    return _staged("error_pattern_baselines", P2 / "run_error_pattern_baselines.py",
                   f"{d}/error_pattern_baselines.yaml", ("all",))


def _error_detector_budget(d: str) -> list[Step]:
    return [Step("error_detector_budget", P2 / "run_error_detector_budget.py",
                 ["--config", f"{d}/error_detector_budget.yaml"])]


def optuna_grids(config: str) -> list[str]:
    """The grids named in a som_optuna.yaml (``optuna.grids``)."""
    raw = yaml.safe_load(resolve(config).read_text(encoding="utf-8")) or {}
    return [str(grid) for grid in raw["optuna"]["grids"]]


def _som_optuna(d: str) -> list[Step]:
    config = f"{d}/som_optuna.yaml"
    grids = optuna_grids(config)
    search = [Step(f"som_optuna: search {g}", OPTUNA / "optuna_grid_tuning.py", ["--config", config, "--grid", g])
              for g in grids]
    confirm = [Step(f"som_optuna: confirmation {g}", OPTUNA / "optuna_confirmation.py",
                    ["--config", config, "--grid", g]) for g in grids]
    summary = Step("som_optuna: summary table", OPTUNA / "optuna_grid_tuning.py", ["--config", config, "--summarize"])
    return [*search, *confirm, summary]


def _feature_layer_ablation(d: str) -> list[Step]:
    return _staged("feature_layer_ablation", P2 / "run_feature_layer_ablation.py",
                   f"{d}/feature_layer_ablation.yaml", ("extract", "som", "compare"))


def _held_out_class_novelty(d: str) -> list[Step]:
    return _staged("held_out_class_novelty", P3 / "run_held_out_class_novelty.py",
                   f"{d}/held_out_class_novelty.yaml", ("run", "compare"))


def _guided_retraining(d: str) -> list[Step]:
    config = f"{d}/retraining.yaml"
    return [
        Step("hard_example_mining", P4 / "run_hard_example_mining.py", ["--config", config]),
        *_staged("targeted_augmentation", P4 / "run_targeted_augmentation.py", config, ("check", "train", "summary")),
        *_staged("curriculum_comparison", P4 / "run_curriculum_comparison.py", config, ("train", "summary")),
    ]


def _guided_retraining_test(d: str) -> list[Step]:
    config = f"{d}/retraining.yaml"
    steps = [
        *_staged("targeted_augmentation", P4 / "run_targeted_augmentation.py", config, ("final_test", "breakdown")),
        *_staged("curriculum_comparison", P4 / "run_curriculum_comparison.py", config, ("final_test", "breakdown")),
        *_staged("som_reanalysis", P4 / "run_som_reanalysis.py", f"{d}/som_reanalysis.yaml",
                 ("embeddings", "train_som", "analyze")),
    ]
    if resolve(f"{d}/som_reanalysis_msp_arm.yaml").is_file():   # the secondary run on the max-softmax arm
        steps += _staged("som_reanalysis (max-softmax arm)", P4 / "run_som_reanalysis.py",
                         f"{d}/som_reanalysis_msp_arm.yaml", ("embeddings", "train_som", "analyze"))
    steps += _staged("novelty_after_retraining", P4 / "run_novelty_after_retraining.py",
                     f"{d}/novelty_after_retraining.yaml", ("score", "compare"))
    return steps


# ---- where each step's results go (read with the step's own settings loader) -------------------


def _first(settings: object) -> object:
    return settings[0] if isinstance(settings, tuple) else settings


def _baseline_results(d: str) -> list[Path]:
    from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.baselines.settings import (
        load_baseline_settings,
    )
    return [_first(load_baseline_settings(f"{d}/error_pattern_baselines.yaml", REPO_ROOT)).output_dir]


def _budget_results(d: str) -> list[Path]:
    from src.v1_mnist.component.p2_som_training_visualization.error_pattern_analysis.baselines.budget_stage import (
        load_budget_settings,
    )
    return [load_budget_settings(f"{d}/error_detector_budget.yaml", REPO_ROOT).output_dir]


def _optuna_results(d: str) -> list[Path]:
    raw = yaml.safe_load(resolve(f"{d}/som_optuna.yaml").read_text(encoding="utf-8")) or {}
    return [resolve(raw["paths"]["summary_table"])]


def _ablation_results(d: str) -> list[Path]:
    from src.v1_mnist.component.p2_som_training_visualization.som_nnsom_training.layer_ablation.settings import (
        load_ablation_settings,
    )
    return [load_ablation_settings(f"{d}/feature_layer_ablation.yaml", REPO_ROOT).output_dir]


def _held_out_results(d: str) -> list[Path]:
    from src.v1_mnist.component.p3_novelty_detection.held_out_class_novelty.settings import (
        load_held_out_settings,
    )
    return [load_held_out_settings(f"{d}/held_out_class_novelty.yaml", REPO_ROOT).output_dir]


def _experiments(d: str) -> list[object]:
    from src.v1_mnist.component.utils.retraining_settings import (
        load_experiment_settings,
    )
    return [load_experiment_settings(section, f"{d}/retraining.yaml", REPO_ROOT)
            for section in ("augmentation", "curriculum")]


def _retraining_results(d: str) -> list[Path]:
    from src.v1_mnist.component.p4_guided_retraining.hard_example_mining.settings import (
        load_pool_settings,
    )
    pools = _first(load_pool_settings(f"{d}/retraining.yaml", REPO_ROOT)).output_dir
    return [pools, *(experiment.output_dir for experiment in _experiments(d))]


def _test_results(d: str) -> list[Path]:
    from src.v1_mnist.component.p4_guided_retraining.novelty_after_retraining.settings import (
        load_novelty_settings,
    )
    from src.v1_mnist.component.p4_guided_retraining.som_reanalysis.settings import (
        load_reanalysis_settings,
    )
    reanalysis = [f"{d}/som_reanalysis.yaml"]
    if resolve(f"{d}/som_reanalysis_msp_arm.yaml").is_file():
        reanalysis.append(f"{d}/som_reanalysis_msp_arm.yaml")
    return [
        *(experiment.output_dir / "test_results.json" for experiment in _experiments(d)),
        *(load_reanalysis_settings(config, REPO_ROOT).output_dir for config in reanalysis),
        load_novelty_settings(f"{d}/novelty_after_retraining.yaml", REPO_ROOT).output_dir,
    ]


PHASE2_STUDY_STEPS = (
    StudyStep("error_pattern_baselines", ("error_pattern_baselines.yaml",), _error_pattern_baselines, _baseline_results),
    StudyStep("error_detector_budget", ("error_detector_budget.yaml",), _error_detector_budget, _budget_results),
    StudyStep("som_optuna", ("som_optuna.yaml",), _som_optuna, _optuna_results),
    StudyStep("feature_layer_ablation", ("feature_layer_ablation.yaml",), _feature_layer_ablation, _ablation_results),
)
LATER_STUDY_STEPS = (
    StudyStep("held_out_class_novelty", ("held_out_class_novelty.yaml",), _held_out_class_novelty, _held_out_results),
    StudyStep("guided_retraining", ("retraining.yaml",), _guided_retraining, _retraining_results),
    StudyStep("guided_retraining_test",
              ("retraining.yaml", "som_reanalysis.yaml", "novelty_after_retraining.yaml"), _guided_retraining_test,
              _test_results),
)
STUDY_STEPS = {step.name: step for step in (*PHASE2_STUDY_STEPS, *LATER_STUDY_STEPS)}
