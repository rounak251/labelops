# Presenting LabelOps in an interview

Use this guide after running the project, inspecting its code, and reproducing the report. Describe the work you personally understand and can explain. The repository is a portfolio project; it is not evidence of employment, production users, or measured business impact.

## A concise project pitch

“LabelOps is a local ML workbench that connects uncertainty-based example selection with multi-reviewer labeling and explicit retraining. I used a small digits dataset to make the entire workflow reproducible, including train-only preprocessing, annotation revisions, model provenance, and a controlled entropy-versus-random benchmark. The project lets me discuss both model evaluation and the engineering needed to maintain trustworthy training labels.”

Adapt the first-person wording to your actual contribution and understanding. Be ready to explain the code rather than memorize this paragraph.

## A five-minute demonstration

| Time | Demonstration | Point to explain |
| --- | --- | --- |
| First minute | Show the queue and one uncertain image | Entropy ranks uncertainty, not guaranteed label value |
| Second minute | Submit two votes, including a disagreement | Annotation policy is explicit; repeated votes from one reviewer do not count twice |
| Third minute | Revise a vote in Label Quality and retrain | Audit history, current consensus, and model state have different lifecycles |
| Fourth minute | Inspect validation metrics and provenance | Explain which rows were fitted and what confidence does and does not mean |
| Fifth minute | Open the benchmark report and one relevant test | Compare equal budgets and identify the simulated-oracle assumption |

If you simulate both reviewers yourself, say so during the demonstration. Use the recorded report for the experiment and the app for the review workflow; they answer different questions.

## Technical talking points

**Why logistic regression?** A fast, inspectable baseline keeps the experiment focused on acquisition and label workflow. A more complex classifier could be a later comparison, but would introduce another variable. Explain the actual solver and regularization in `model.py`.

**Why put scaling in a pipeline?** The scaler must learn from exactly the fitting rows. Training it on validation, test, or the full unlabeled pool would violate the experiment's train-only fitting rule.

**Why compare against random sampling?** Random acquisition is a simple baseline that costs the same number of labels. Without it, a rising learning curve could reflect the additional labels rather than a better selection rule.

**What does entropy measure?** It summarizes how spread out the predicted class probabilities are. It can favor ambiguous or noisy images, and its usefulness depends on the current model and data. It does not directly measure expected accuracy improvement.

**How does consensus work?** Each reviewer has one current vote per sample. At least two reviewers are required, and more than half must support one label. Revisions append events, preserving the history while updating the current view.

**Why preserve provenance?** A model can remain fitted on an earlier consensus after a vote changes. Recording fit IDs and label sources makes that state visible and helps reproduce the model.

**Why validate saved models?** A dataset-compatible cache may still use outdated hyperparameters or dependencies. LabelOps checks the model configuration, package versions, and estimator parameters before restoring a fitted artifact.

**Why check persisted annotations?** Input validation on HTTP requests is only one boundary. A prepopulated annotation database must also satisfy the training-pool membership rule before it can contribute to fitting.

**Why separate validation and test?** Validation supports diagnostics during development. The test set is reserved for explicit final evaluation and never enters acquisition or training. Selecting settings after repeated test inspection would weaken that separation.

**Why report probability diagnostics?** Accuracy does not describe whether confident predictions are reliable. Log loss, Brier score, and calibration bins reveal different issues, although this project does not fit a probability calibrator.

**What would change for a real annotation team?** The UI already hides suggestions by default. Enforced independent review, an abstain option, adjudication, authentication, shared model-state coordination, and measured reviewer time would still matter. Majority voting alone is insufficient evidence of label correctness.

## Resume bullet templates

Use only statements you can support by the code, a run, or your actual contribution. Do not paste unfilled placeholders into a resume.

- Built a local human-in-the-loop classification workbench with Streamlit, FastAPI, and scikit-learn, connecting uncertainty-ranked review, multi-reviewer consensus, auditable label revisions, and explicit retraining.
- Implemented a reproducible entropy-versus-random acquisition benchmark across three seeded splits with equal label budgets, train-only preprocessing, and separate validation and test evaluation.
- Added behavioral tests for data isolation, annotation consensus, model provenance, and API validation; documented experimental assumptions and dataset limitations in a model card.
- **Optional measured result:** Compared entropy and random acquisition at **[label budget]** labels, obtaining **[metric and value]** on **[validation or test split]** under **[seed aggregation]**; published the reproducible configuration and report.

The first three are contribution templates, not certifications of authorship. If you used AI assistance, be candid when asked and explain the choices you verified or changed. Add a performance number only after reproducing it and stating its scope. Do not claim percentage label savings from this oracle simulation, real users, cloud deployment, or production adoption.

The recorded example comparison is 95.19% versus 93.06% mean final test accuracy at 300 simulated labels across three seeded runs. The +2.13 percentage-point difference describes this small controlled experiment; it does not establish statistical significance or real annotation savings.

## Useful follow-up experiments

1. Add margin sampling and compare it at the same acquisition budget.
2. Measure sensitivity to initial seed size while keeping the final test protocol fixed.
3. Introduce simulated noisy labels and show when uncertainty sampling becomes less useful.
4. Add an abstain/adjudication workflow, keep model suggestions hidden, and evaluate a small real review exercise.
5. Replace digits with a dataset that has group identifiers and use group-aware evaluation.

Choose one extension, state its hypothesis before running it, and preserve the baseline report. A carefully explained negative result is useful evidence of engineering judgment.
