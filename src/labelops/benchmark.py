"""Paired, reproducible oracle-label simulations with a sealed final test set."""

from __future__ import annotations

import csv
import json
import os
import platform
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import sklearn

from labelops.data import dataset_provenance, load_dataset
from labelops.metrics import classification_metrics
from labelops.model import MODEL_CONFIG, fit_model, rank_uncertainty

STRATEGIES = ("random", "entropy")
SCORE_NAMES = ("accuracy", "macro_f1", "log_loss", "ece", "brier")


def _mean_std(values: list[float]) -> dict:
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0))}


def _summarize(runs: list[dict], rounds: int) -> dict:
    curves = []
    for strategy in STRATEGIES:
        for round_index in range(rounds + 1):
            points = [run["strategies"][strategy]["learning_curve"][round_index] for run in runs]
            curves.append(
                {
                    "strategy": strategy,
                    "round": round_index,
                    "labeled_count": points[0]["labeled_count"],
                    **{
                        f"validation_{score}": _mean_std(
                            [point["validation_metrics"][score] for point in points]
                        )
                        for score in SCORE_NAMES
                    },
                }
            )
    final_test = {
        strategy: {
            score: _mean_std([run["strategies"][strategy]["test_metrics"][score] for run in runs])
            for score in SCORE_NAMES
        }
        for strategy in STRATEGIES
    }
    paired_deltas = [
        {
            "seed": run["seed"],
            "accuracy_delta_entropy_minus_random": (
                run["strategies"]["entropy"]["test_metrics"]["accuracy"]
                - run["strategies"]["random"]["test_metrics"]["accuracy"]
            ),
        }
        for run in runs
    ]
    return {
        "learning_curve": curves,
        "final_test": final_test,
        "paired_test_accuracy_deltas": paired_deltas,
        "paired_test_accuracy_delta": _mean_std(
            [point["accuracy_delta_entropy_minus_random"] for point in paired_deltas]
        ),
        "dispersion": "Population standard deviation across seed runs; not a confidence interval.",
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = ["seed", "strategy", "round", "labeled_count"] + [
        f"validation_{score}" for score in SCORE_NAMES
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pct(summary: dict) -> str:
    return f"{summary['mean']:.1%} ± {summary['std']:.1%}"


def _write_markdown(path: Path, report: dict) -> None:
    experiment = report["experiment"]
    aggregate = report["aggregate"]
    curves = {(point["strategy"], point["round"]): point for point in aggregate["learning_curve"]}
    lines = [
        "# LabelOps benchmark",
        "",
        "This is an educational **oracle-label simulation**, not a study of human annotation quality or time savings.",
        "",
        f"Seeds: `{', '.join(map(str, experiment['seeds']))}`. Each seed has a stratified 60/20/20 sample split, "
        f"{experiment['initial_labeled_count']} initial training labels, and {experiment['rounds']} batches of "
        f"{experiment['batch_size']} simulated labels. Strategies share initial labels and equal budgets within each seed.",
        "",
        "The fixed CPU model is StandardScaler followed by logistic regression (C=1, lbfgs, max_iter=2000, tol=1e-4). "
        "Scaling and classification fit labeled training rows only. Entropy acquisition sees predictions on the remaining "
        "training pool; validation and test rows are excluded from both fitting and acquisition.",
        "",
        "## Validation learning curve",
        "",
        "| Label budget | Random accuracy | Entropy accuracy | Random macro F1 | Entropy macro F1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for round_index in range(experiment["rounds"] + 1):
        random = curves[("random", round_index)]
        entropy = curves[("entropy", round_index)]
        lines.append(
            f"| {random['labeled_count']} | {_pct(random['validation_accuracy'])} | {_pct(entropy['validation_accuracy'])} "
            f"| {_pct(random['validation_macro_f1'])} | {_pct(entropy['validation_macro_f1'])} |"
        )
    lines += [
        "",
        "Values are means ± population standard deviation across seeds. Shaded bands in the SVG show this dispersion, "
        "not confidence intervals. Both the model parameters and final budget are fixed before running this experiment.",
        "",
        "## Sealed final test evaluation",
        "",
        "Test labels are accessed once per strategy per seed, after the final fixed training round. "
        "They do not determine hyperparameters, stopping, acquisition, or the reported budget.",
        "",
        "| Strategy | Accuracy | Macro F1 | Log loss | ECE | Multiclass Brier |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy in STRATEGIES:
        scores = aggregate["final_test"][strategy]
        lines.append(
            f"| {strategy.capitalize()} | {_pct(scores['accuracy'])} | {_pct(scores['macro_f1'])} "
            f"| {scores['log_loss']['mean']:.3f} | {scores['ece']['mean']:.3f} | {scores['brier']['mean']:.3f} |"
        )
    delta = aggregate["paired_test_accuracy_delta"]
    lines += [
        "",
        f"Paired test accuracy change (entropy minus random): **{delta['mean'] * 100:+.2f} percentage points** "
        f"on average (standard deviation {delta['std'] * 100:.2f} points). This small benchmark does not establish "
        "statistical significance or guarantee that entropy sampling beats random acquisition on another dataset.",
        "",
        "## Data provenance and limits",
        "",
        "The 1,797 bundled 8×8 digit images are loaded directly from `sklearn.datasets.load_digits`, "
        "which contains the test partition of the UCI Optical Recognition of Handwritten Digits dataset. "
        "This project resplits those samples; it does not reproduce the original UCI evaluation protocol. "
        "Writer identity is unavailable in this loader, so the splits are not writer-disjoint.",
        "",
        "- [scikit-learn loader documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)",
        "- [Original UCI dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)",
        "- Simulated labels are the built-in ground truth, not independent human votes.",
        "- ECE uses ten equal-width top-confidence bins; confidence is not guaranteed calibrated.",
        "- Multiclass Brier is the mean sum of squared class probability errors (range 0–2).",
        "- Selection IDs, training IDs, split IDs, dataset fingerprints, and package versions are in `report.json`.",
        "- Randomness is seeded and fitting uses one numerical thread. Reproduce within the pinned environment; "
        "numerical libraries can change results across versions or platforms.",
        "",
        "Generated files: `report.json`, `learning_curve.csv`, `learning_curve.svg`, `confusion_matrix.svg`, `report.md`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_plots(output_dir: Path, report: dict) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "labelops-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    styles = {"random": ("#64748b", "Random sampling"), "entropy": ("#0891b2", "Entropy sampling")}
    with plt.rc_context(
        {
            "svg.hashsalt": "labelops",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
        for strategy in STRATEGIES:
            points = [
                point
                for point in report["aggregate"]["learning_curve"]
                if point["strategy"] == strategy
            ]
            budgets = np.array([point["labeled_count"] for point in points])
            means = np.array([point["validation_accuracy"]["mean"] for point in points])
            stds = np.array([point["validation_accuracy"]["std"] for point in points])
            color, name = styles[strategy]
            ax.plot(budgets, means, marker="o", color=color, linewidth=2.5, label=name)
            ax.fill_between(
                budgets,
                np.maximum(means - stds, 0),
                np.minimum(means + stds, 1),
                color=color,
                alpha=0.15,
            )
        ax.set_xlabel("Training label budget (simulated oracle labels)")
        ax.set_ylabel("Validation accuracy")
        ax.set_title(f"Paired active-learning simulation · {len(report['runs'])} seeds")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False)
        ax.set_ylim(max(0.0, ax.get_ylim()[0] - 0.02), min(1.0, ax.get_ylim()[1] + 0.02))
        fig.savefig(output_dir / "learning_curve.svg", metadata={"Date": None})
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
        for ax, strategy in zip(axes, STRATEGIES, strict=True):
            matrix = np.sum(
                [
                    np.array(run["strategies"][strategy]["test_metrics"]["confusion_matrix"])
                    for run in report["runs"]
                ],
                axis=0,
            )
            denominator = matrix.sum(axis=1, keepdims=True)
            normalized = np.divide(
                matrix, denominator, out=np.zeros_like(matrix, dtype=float), where=denominator != 0
            )
            image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
            for row in range(10):
                for column in range(10):
                    if normalized[row, column] > 0:
                        ax.text(
                            column,
                            row,
                            f"{normalized[row, column]:.2f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="white" if normalized[row, column] > 0.55 else "#1e293b",
                        )
            ax.set_xticks(range(10))
            ax.set_yticks(range(10))
            ax.set_xlabel("Predicted digit")
            ax.set_ylabel("True digit")
            ax.set_title(styles[strategy][1])
        fig.colorbar(image, ax=axes, shrink=0.75, label="Fraction within true class")
        fig.suptitle("Final sealed test confusion · aggregated seed runs")
        fig.savefig(output_dir / "confusion_matrix.svg", metadata={"Date": None})
        plt.close(fig)
    for name in ("learning_curve.svg", "confusion_matrix.svg"):
        plot = output_dir / name
        plot.write_text(
            "\n".join(line.rstrip() for line in plot.read_text(encoding="utf-8").splitlines())
            + "\n",
            encoding="utf-8",
        )


def run_benchmark(
    output_dir: str | Path,
    seeds: tuple[int, ...] = (7, 21, 42),
    rounds: int = 5,
    batch_size: int = 40,
) -> dict:
    """Compare fixed entropy and random acquisition with paired label budgets.

    Oracle training-pool labels simulate acquisitions only. Each final model is
    evaluated on that seed's fixed test split once, after the predetermined
    rounds. The test split never enters training, acquisition, or validation.
    """
    seeds = tuple(int(seed) for seed in seeds)
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(seed < 0 or seed >= 2**32 for seed in seeds)
    ):
        raise ValueError("Seeds must be distinct integers between 0 and 2**32 - 1.")
    if (
        not isinstance(rounds, int)
        or rounds < 0
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("Rounds must be nonnegative and batch size must be positive integers.")
    runs = []
    csv_rows = []
    for seed in seeds:
        dataset = load_dataset(seed)
        if rounds * batch_size > len(dataset.pool_ids):
            raise ValueError("The requested label budget exceeds the training pool.")
        run = {
            "seed": seed,
            "dataset": dataset_provenance(dataset),
            "initial_labeled_ids": dataset.seed_ids.tolist(),
            "validation_ids": dataset.validation_ids.tolist(),
            "test_ids": dataset.test_ids.tolist(),
            "strategies": {},
        }
        for strategy in STRATEGIES:
            labeled_ids = dataset.seed_ids.copy()
            pool_ids = dataset.pool_ids.copy()
            rng = np.random.default_rng(seed)
            history = []
            acquisitions = []
            for round_index in range(rounds + 1):
                model = fit_model(dataset.X[labeled_ids], dataset.y[labeled_ids], seed=seed)
                validation_metrics = classification_metrics(
                    dataset.y[dataset.validation_ids],
                    model.predict_proba(dataset.X[dataset.validation_ids]),
                )
                history.append(
                    {
                        "round": round_index,
                        "labeled_count": int(len(labeled_ids)),
                        "validation_metrics": validation_metrics,
                    }
                )
                csv_rows.append(
                    {
                        "seed": seed,
                        "strategy": strategy,
                        "round": round_index,
                        "labeled_count": len(labeled_ids),
                        **{
                            f"validation_{score}": validation_metrics[score]
                            for score in SCORE_NAMES
                        },
                    }
                )
                if round_index == rounds:
                    break
                if strategy == "entropy":
                    selected_ids = rank_uncertainty(
                        model.predict_proba(dataset.X[pool_ids]), pool_ids
                    )[:batch_size]
                else:
                    selected_ids = rng.choice(pool_ids, size=batch_size, replace=False)
                acquisitions.append({"round": round_index + 1, "sample_ids": selected_ids.tolist()})
                labeled_ids = np.sort(np.concatenate([labeled_ids, selected_ids]))
                pool_ids = pool_ids[~np.isin(pool_ids, selected_ids)]
            test_metrics = classification_metrics(
                dataset.y[dataset.test_ids], model.predict_proba(dataset.X[dataset.test_ids])
            )
            run["strategies"][strategy] = {
                "learning_curve": history,
                "acquisitions": acquisitions,
                "final_train_ids": labeled_ids.tolist(),
                "test_metrics": test_metrics,
            }
        runs.append(run)
    report = {
        "schema_version": 1,
        "experiment": {
            "name": "Paired entropy versus random oracle-label simulation",
            "seeds": list(seeds),
            "strategies": list(STRATEGIES),
            "rounds": rounds,
            "batch_size": batch_size,
            "initial_labeled_count": 100,
            "final_label_budget": 100 + rounds * batch_size,
            "acquisition": "Training pool only; descending entropy with ID tie breaks, or seeded uniform random sampling.",
            "evaluation_protocol": "Validation curves at fixed budgets; one final sealed test evaluation per strategy per seed.",
            "oracle_disclosure": "Ground-truth training labels simulate annotation; no human annotation gains are measured.",
            "metrics": {
                "ece": "10 equal-width top-label confidence bins; weighted absolute confidence-minus-accuracy gap",
                "brier": "mean sum of squared probability errors across 10 classes (range 0..2)",
                "confusion_matrix": "rows=true digit, columns=predicted digit",
            },
        },
        "model": dict(MODEL_CONFIG),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": version("scipy"),
            "joblib": version("joblib"),
            "threadpoolctl": version("threadpoolctl"),
            "matplotlib": version("matplotlib"),
            "operating_system": platform.system(),
            "architecture": platform.machine(),
        },
        "runs": runs,
        "aggregate": _summarize(runs, rounds),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "learning_curve.csv", csv_rows)
    _write_markdown(output_dir / "report.md", report)
    _write_plots(output_dir, report)
    return report
