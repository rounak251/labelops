# LabelOps benchmark

This is an educational **oracle-label simulation**, not a study of human annotation quality or time savings.

Seeds: `7, 21, 42`. Each seed has a stratified 60/20/20 sample split, 100 initial training labels, and 5 batches of 40 simulated labels. Strategies share initial labels and equal budgets within each seed.

The fixed CPU model is StandardScaler followed by logistic regression (C=1, lbfgs, max_iter=2000, tol=1e-4). Scaling and classification fit labeled training rows only. Entropy acquisition sees predictions on the remaining training pool; validation and test rows are excluded from both fitting and acquisition.

## Validation learning curve

| Label budget | Random accuracy | Entropy accuracy | Random macro F1 | Entropy macro F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 88.1% ± 3.1% | 88.1% ± 3.1% | 88.1% ± 3.0% | 88.1% ± 3.0% |
| 140 | 89.1% ± 2.4% | 88.7% ± 1.8% | 89.2% ± 2.3% | 88.7% ± 1.7% |
| 180 | 89.9% ± 2.3% | 90.3% ± 1.4% | 89.9% ± 2.2% | 90.3% ± 1.3% |
| 220 | 91.1% ± 1.5% | 93.9% ± 0.5% | 91.1% ± 1.4% | 93.9% ± 0.4% |
| 260 | 91.7% ± 0.1% | 95.4% ± 1.1% | 91.7% ± 0.2% | 95.3% ± 1.1% |
| 300 | 93.0% ± 0.8% | 96.0% ± 0.3% | 93.0% ± 0.8% | 96.0% ± 0.3% |

Values are means ± population standard deviation across seeds. Shaded bands in the SVG show this dispersion, not confidence intervals. Both the model parameters and final budget are fixed before running this experiment.

## Sealed final test evaluation

Test labels are accessed once per strategy per seed, after the final fixed training round. They do not determine hyperparameters, stopping, acquisition, or the reported budget.

| Strategy | Accuracy | Macro F1 | Log loss | ECE | Multiclass Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 93.1% ± 1.0% | 93.1% ± 1.0% | 0.234 | 0.033 | 0.109 |
| Entropy | 95.2% ± 0.5% | 95.2% ± 0.5% | 0.159 | 0.033 | 0.072 |

Paired test accuracy change (entropy minus random): **+2.13 percentage points** on average (standard deviation 0.92 points). This small benchmark does not establish statistical significance or guarantee that entropy sampling beats random acquisition on another dataset.

## Data provenance and limits

The 1,797 bundled 8×8 digit images are loaded directly from `sklearn.datasets.load_digits`, which contains the test partition of the UCI Optical Recognition of Handwritten Digits dataset. This project resplits those samples; it does not reproduce the original UCI evaluation protocol. Writer identity is unavailable in this loader, so the splits are not writer-disjoint.

- [scikit-learn loader documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- [Original UCI dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)
- Simulated labels are the built-in ground truth, not independent human votes.
- ECE uses ten equal-width top-confidence bins; confidence is not guaranteed calibrated.
- Multiclass Brier is the mean sum of squared class probability errors (range 0–2).
- Selection IDs, training IDs, split IDs, dataset fingerprints, and package versions are in `report.json`.
- Randomness is seeded and fitting uses one numerical thread. Reproduce within the pinned environment; numerical libraries can change results across versions or platforms.

Generated files: `report.json`, `learning_curve.csv`, `learning_curve.svg`, `confusion_matrix.svg`, `report.md`.
