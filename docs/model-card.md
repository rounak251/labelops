# Model card

## Purpose

LabelOps is an educational ML engineering and annotation-workflow demonstration. Its classifier predicts one of ten handwritten digit classes (`0` through `9`) from 64 pixel features. The workbench lets reviewers label uncertain training examples and makes retraining and evaluation explicit.

The intended audience is developers learning about active learning, label quality, reproducible experiments, and serving a small ML system. The model is not evaluated for document processing, handwriting recognition outside this dataset, or decisions affecting people.

## Data and provenance

The data comes from [`sklearn.datasets.load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html): 1,797 examples, 8×8 images, ten classes, and pixel values from 0 to 16. This bundled data is the original UCI dataset's test portion, not all 5,620 examples in the source dataset. LabelOps creates a new stratified sample split inside the bundled portion.

Original dataset: E. Alpaydin and C. Kaynak (1998), [*Optical Recognition of Handwritten Digits*, UCI](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), [DOI: 10.24432/C50P49](https://doi.org/10.24432/C50P49). UCI lists [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) for the dataset.

The application retains original row indices as stable sample IDs and records a dataset fingerprint. No external image collection, personal dataset, or pretrained model is required.

## Model and fitting

The classifier is a scikit-learn pipeline containing `StandardScaler` and `LogisticRegression`. Both components are refitted on the same selected training rows. The fixed classifier configuration is `C=1.0`, `solver="lbfgs"`, `max_iter=2000`, `tol=1e-4`, and `random_state` equal to the split seed. Fitting uses one numerical thread on the CPU.

In the review workbench, the first fit uses 100 stratified seed examples from the training split. Subsequent fits use those seeds plus current reviewer-consensus labels. A consensus label requires at least two distinct reviewer identifiers and a strict majority of their latest votes. The original dataset labels of pool examples do not replace reviewer votes during workbench retraining.

The service rejects persisted annotation records outside the training pool at startup and before retraining. This preserves the fitting boundary even when a workspace has been populated through the lower-level annotation store.

Model provenance records fitting sample IDs, label sources, the model configuration, package versions, model version, and dataset identity. The local cache validates dataset, configuration, dependency metadata, and actual estimator parameters before reuse; an incompatible cache triggers a refit from the annotation audit. Only load workspace artifacts that you created or otherwise trust, because serialized Python model files are executable formats.

## Evaluation protocol

| Partition | Role | Used for fitting or acquisition? |
| --- | --- | --- |
| Training: approximately 60% | Initial seed labels and review/acquisition pool | Yes, but fitting uses labeled rows only |
| Validation: approximately 20% | Learning curves and development diagnostics | No |
| Test: approximately 20% | Explicit final evaluation | No |

Splits are stratified by digit class and fixed by seed. The test set remains outside fitting and acquisition. The dashboard ordinarily reports validation results; test evaluation is an explicit operation. Repeatedly inspecting test scores to make design decisions would weaken the final evaluation boundary.

These are sample-level splits. The bundled loader does not provide writer IDs for a group-based evaluation, so the project does not claim writer-disjoint generalization. The three seeded benchmark runs measure sensitivity to sample splitting and acquisition randomness; they are not a population-level uncertainty estimate.

## Acquisition benchmark

The default experiment runs entropy and random sampling for seeds `7`, `21`, and `42`. Both strategies start with the same 100 labels for each seed and add 40 labels per round for five rounds, ending at 300 labels. Each strategy uses the same model configuration and its own remaining training pool.

Entropy is calculated from the model's ten predicted probabilities:

```text
H(p) = - sum over classes c of p(c) × log(p(c))
```

Higher entropy places an example earlier in the queue. Equal scores are resolved by stable sample ID. This prioritizes uncertainty, which may also reflect noise or a poorly fitted model; it is not a direct estimate of an example's value.

The offline benchmark uses the dataset's original labels as a simulated oracle after selection. It assumes immediate, correct labels and equal cost per example. Its results support a comparison of acquisition rules in this controlled dataset, not claims about actual human annotation savings. Real review time, annotator mistakes, disagreement, and changing data are outside that experiment.

## Metrics and measured results

| Metric | Interpretation | Preferred direction |
| --- | --- | --- |
| Accuracy | Fraction of correctly predicted examples | Higher |
| Macro F1 | Mean F1 across all ten classes, weighting classes equally | Higher |
| Log loss | Penalty for probability assigned to the true class | Lower |
| Expected calibration error (ECE) | Sample-weighted absolute difference between top-class confidence and accuracy across ten equal-width confidence bins | Lower |
| Multiclass Brier score | Mean per-example sum of squared differences between predicted probabilities and the one-hot true label; range 0–2 | Lower |
| Confusion matrix and per-class metrics | Classes and error patterns hidden by aggregate scores | Inspect |

Probability confidence is a model output, not a guarantee of correctness. No separate calibration procedure is fitted. ECE depends on its binning scheme and can look reassuring even when important subgroups are poorly calibrated. Accuracy and macro F1 describe different aspects of performance; neither captures annotation cost.

At the final 300-label budget, test accuracy averaged 95.19% for entropy sampling and 93.06% for random sampling across three seeds. Population standard deviations were 0.47 and 0.99 percentage points respectively. The reference run used Python 3.13 with constrained dependencies. Read the generated [benchmark report](../reports/report.md), [raw JSON](../reports/report.json), and [learning-curve CSV](../reports/learning_curve.csv) for measured results and configuration. Report numbers with their strategy, seed or aggregation, label budget, split, and environment. Do not substitute validation performance for test performance.

## Limitations

- The images are tiny, centered, preprocessed digits rather than raw handwriting or photographs.
- The sample count is small, and results can vary with the initial labels and split.
- Stratification uses known benchmark class labels when constructing the initial split and seed set; this is a controlled educational setup.
- The UI hides prediction suggestions by default; reviewers can reveal them using a sidebar toggle. Revealed suggestions can anchor judgment, and the API queue always includes predictions.
- Reviewer identifiers are self-reported. Different identifiers do not verify different people.
- Majority consensus is a simple policy. It does not model expertise, detect coordinated mistakes, or guarantee a correct label.
- A local workflow lacks access control, multi-user deployment hardening, and production monitoring.
- Separate UI/API processes have independent cached fitted models, even when annotations share a workspace. Model changes do not automatically synchronize across processes.
- The experiment does not measure fairness across writer demographics, domain shift, real reviewer productivity, or production reliability.

## Change control

If changing the dataset, model, split, acquisition rule, or label policy, regenerate the benchmark and update this card. Preserve the distinction between observed results and proposed extensions. A new domain needs representative data, appropriate grouped or temporal evaluation, annotation guidelines, and a fresh error analysis before any deployment claim.
