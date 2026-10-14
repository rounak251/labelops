# Annotation guide

The task is to label each displayed handwritten image with one digit from `0` to `9`. The image is an 8×8 intensity grid from the bundled digits dataset. The UI hides the model suggestion by default. You can reveal it with **Show model suggestion** in the sidebar after forming your own judgment; it is a prediction and may be wrong.

## A review session

1. Use a consistent reviewer identifier. In a real collaboration, each person should use their own identifier.
2. Inspect the image before relying on the prediction. Consider the shape, open or closed loops, and stroke arrangement.
3. Select the most plausible digit and add a short note when the image is ambiguous or the prediction seems wrong.
4. Submit the annotation. A second reviewer should inspect the example independently.
5. Review disputed examples and revise a vote if needed. Retrain after labels reach consensus.

If you cannot assign a defensible digit, leave the example unsubmitted and record the case outside the tool for adjudication. The current schema accepts only digit labels; it has no dedicated abstain or unreadable class. Do not turn an uncertain guess into an apparently confident annotation to clear the queue.

For a solo demonstration, use clearly named demo reviewers to exercise revisions and disagreement. Report that as a simulation. Two identities controlled by one person are not independent human reviews.

## Consensus rules

Only each reviewer's latest vote on an example participates in consensus. At least two reviewers must have voted, and a label must receive a strict majority.

| Current votes | Consensus | Explanation |
| --- | --- | --- |
| Alice: `7` | None | A second reviewer is required |
| Alice: `7`, Bob: `7` | `7` | Both agree |
| Alice: `7`, Bob: `1` | None | Two-way tie |
| Alice: `7`, Bob: `1`, Carol: `7` | `7` | Two of three is a strict majority |
| Alice: `7`, Bob: `1`, Carol: `7`, Dev: `1` | None | No label has more than half |

To revise a vote in the UI, open **Label Quality → Revise one of your saved reviews**, select the example, and submit the replacement label. Changing Alice's vote adds another event to the audit history and replaces Alice's previous vote in the current view. Repeated submissions by one reviewer do not create extra votes.

Consensus examples leave the review queue. Consensus changes affect the training labels on the next explicit retrain; submitting a vote does not immediately change the fitted classifier.

## Useful notes

A useful note records the visual reason for a decision or the reason for changing a vote:

- “Upper loop is closed; lower stroke looks like a 9.”
- “Revised from 1 to 7 after inspecting the horizontal top stroke.”
- “Could be 3 or 8; left side appears open.”

Keep notes short and task-related. Reviewer identifiers and notes are stored locally and exported with the annotation data, so use a handle rather than personal contact details.

## Reading agreement

Agreement counts and pairwise Cohen's kappa help locate inconsistent review. They should be read alongside the number of shared examples. A high score on very few overlapping examples is weak evidence, and consistent reviewers can still share the same mistake. Inspect actual disagreements before changing the policy.

The default hidden-suggestion mode helps reviewers form their own judgments. Revealing a model prediction may encourage anchoring. For a study of human agreement, keep suggestions hidden and have people label independently; the API queue still includes predictions. The current app demonstrates the workflow; it does not establish the quality of a human annotation team.

## Export and reproducibility

Export annotations with:

```bash
labelops export --workspace .labelops --output annotations.csv
```

The CSV contains each reviewer's latest vote on each sample, including its event details. Preserve `annotations.sqlite3` when you need the full revision history, and preserve the workspace model bundle for fit provenance. The benchmark reports are a separate oracle-label experiment, so do not describe those records as annotations made by people.

Separate UI and API processes do not automatically synchronize their cached models. If another process retrains against a shared workspace, restart or retrain your process before comparing model results.
