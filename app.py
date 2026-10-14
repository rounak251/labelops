"""Run with `streamlit run app.py` after installing LabelOps."""

import json
import os
from pathlib import Path
from threading import RLock

import numpy as np
import pandas as pd
import streamlit as st

from labelops.service import Workbench

st.set_page_config(page_title="LabelOps · ML Training Workbench", page_icon="◈", layout="wide")
st.markdown(
    """<style>
    .block-container {max-width: 1280px; padding-top: 4.5rem;}
    h1 {letter-spacing: -0.045em;}
    div[data-testid="stMetric"] {background:#17212f; border:1px solid #263447;
        padding:18px; border-radius:14px;}
    .eyebrow {color:#80e9c8; font-size:12px; letter-spacing:0.16em; font-weight:700;}
    .hero {color:#9badc6; font-size:18px; max-width:800px; margin-bottom:24px;}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def workbench_at(path: str):
    return Workbench(Path(path)), RLock()


with st.sidebar:
    st.markdown("### ◈ LabelOps")
    st.caption("A better training loop starts with better labels.")
    workspace_path = st.text_input(
        "Workspace", value=os.environ.get("LABELOPS_WORKSPACE", ".labelops")
    )
    reviewer = st.text_input("Reviewer name", value="reviewer-1", max_chars=80).strip()
    st.caption("Use a distinct name for each person. Labels need agreement from two reviewers.")
    show_prediction = st.toggle("Show model suggestion", value=False)
    st.caption("Review the image first to reduce anchoring on the model's answer.")
    st.divider()
    st.markdown("**Demo dataset**  \nHandwritten digits · 8 × 8 pixels")
    st.caption("Local CPU training. No API keys. Built for learning and portfolio demonstration.")

if not workspace_path.strip():
    st.info("Enter a workspace directory to start.")
    st.stop()

workbench, lock = workbench_at(str(Path(workspace_path).resolve()))
with lock:
    summary = workbench.summary()

st.markdown('<div class="eyebrow">HUMAN-IN-THE-LOOP MACHINE LEARNING</div>', unsafe_allow_html=True)
st.title("Better labels. Better models.")
st.markdown(
    '<div class="hero">Prioritize uncertain examples, collect independent reviews, '
    "and turn reviewer consensus into reproducible model improvements.</div>",
    unsafe_allow_html=True,
)

overview, review_tab, quality_tab, benchmark_tab = st.tabs(
    ["Overview", "Review Queue", "Label Quality", "Benchmark"]
)

with overview:
    validation = summary["validation_metrics"]
    agreement = summary["agreement"]
    columns = st.columns(4)
    columns[0].metric("Approved labels", summary["labeled_count"])
    columns[1].metric("Validation macro F1", f"{validation['macro_f1']:.1%}")
    columns[2].metric("Reviewer consensus", agreement["consensus_count"])
    columns[3].metric("Awaiting retraining", summary["pending_consensus_count"])
    st.caption(
        f"Model version {summary['model_version']} · trained on {summary['fitted_labeled_count']} labels · "
        "validation is held out from training · "
        "confidence values are estimates, not guarantees"
    )
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Per-class performance")
        per_class = validation.get("per_class", [])
        frame = (
            pd.DataFrame(per_class).T if isinstance(per_class, dict) else pd.DataFrame(per_class)
        )
        st.dataframe(frame, hide_index=False, width="stretch")
    with right:
        st.subheader("How the loop works")
        st.markdown(
            "1. **Review:** inspect uncertain images independently.\n"
            "2. **Agree:** at least two reviewers establish a majority label.\n"
            "3. **Retrain:** fit on seed labels and reviewer consensus.\n"
            "4. **Measure:** monitor validation quality and label disagreement."
        )
        if st.button("Retrain with consensus labels", type="primary", width="stretch"):
            with st.spinner("Training from the approved labels…"), lock:
                workbench.retrain()
            st.rerun()
    with st.expander("Data boundaries and model provenance"):
        st.json({"dataset_counts": summary["dataset_counts"], "model": summary["model_provenance"]})
        st.caption(
            "Reviewers cannot label the validation or test partitions through this interface."
        )
    with st.expander("Inspect validation mistakes"):
        with lock:
            ids = workbench.dataset.validation_ids
            probabilities = workbench.model.predict_proba(workbench.dataset.X[ids])
            predictions = probabilities.argmax(axis=1)
            mistakes = np.flatnonzero(predictions != workbench.dataset.y[ids])
            order = mistakes[np.argsort(-probabilities[mistakes].max(axis=1))][:6]
        st.caption(
            "The six most confident mistakes on the validation partition; never used to fit this model."
        )
        if len(order):
            image_columns = st.columns(len(order))
            for column, position in zip(image_columns, order):
                sample_id = int(ids[position])
                pixels = (workbench.dataset.X[sample_id].reshape(8, 8) * (255 / 16)).astype(
                    np.uint8
                )
                enlarged = np.repeat(np.repeat(pixels, 14, axis=0), 14, axis=1)
                column.image(enlarged, width=112)
                column.caption(
                    f"Reference {workbench.dataset.y[sample_id]} · predicted {predictions[position]}"
                )
        else:
            st.success("No mistakes in this validation evaluation.")

with review_tab:
    st.subheader("Your next useful label")
    st.caption("Sorted by predictive entropy. Higher entropy means the model is less certain.")
    with lock:
        annotations = workbench.store.annotations()
        queue = workbench.queue(limit=len(workbench.dataset.pool_ids))
    already_reviewed = {row["sample_id"] for row in annotations if row["reviewer"] == reviewer}
    candidates = [row for row in queue if row["id"] not in already_reviewed][:12]
    if not reviewer:
        st.info("Enter your reviewer name in the sidebar to submit labels.")
    elif not candidates:
        st.success(
            "You have reviewed all available examples. Another reviewer can add a second vote."
        )
    else:
        by_id = {row["id"]: row for row in candidates}
        selected_id = st.selectbox(
            "Example",
            list(by_id),
            format_func=lambda key: (
                f"Sample #{key} · {by_id[key]['review_count']} reviews"
                + (" · disagreement" if by_id[key]["disputed"] else "")
            ),
        )
        selected = by_id[selected_id]
        image_column, label_column, signal_column = st.columns([1, 2, 1])
        with image_column:
            pixels = (np.asarray(selected["image"], dtype=float) * (255.0 / 16)).astype(np.uint8)
            enlarged = np.repeat(np.repeat(pixels, 28, axis=0), 28, axis=1)
            st.image(enlarged, caption=f"Sample #{selected_id}", width=224, clamp=True)
        with label_column:
            with st.form("annotation"):
                label = st.radio(
                    "Which digit do you see?", list(range(10)), index=None, horizontal=True
                )
                note = st.text_area(
                    "Review note (optional)",
                    placeholder="Describe ambiguity or the evidence for your label.",
                    max_chars=2000,
                )
                submitted = st.form_submit_button("Save review", type="primary")
            if submitted and label is None:
                st.warning("Choose a digit before saving your review.")
            elif submitted:
                with lock:
                    workbench.annotate(selected_id, reviewer, label, note)
                st.rerun()
        with signal_column:
            st.metric("Uncertainty", f"{selected['entropy']:.3f}")
            st.metric("Independent reviews", selected["review_count"])
            if show_prediction:
                st.metric("Model suggestion", selected["prediction"])
                st.caption(f"Confidence: {selected['confidence']:.1%}")
                st.bar_chart(pd.Series(selected["probabilities"], name="Probability"))
            else:
                st.caption("Model suggestion hidden. Make your own judgement before checking it.")
    st.caption("This view never displays the benchmark's reference label for a review example.")

with quality_tab:
    st.subheader("Independent reviews, visible disagreements")
    agreement = summary["agreement"]
    cols = st.columns(3)
    cols[0].metric("Reviewers", agreement["annotator_count"])
    cols[1].metric("Latest votes", agreement["annotation_count"])
    cols[2].metric("Unresolved disagreements", agreement["disputed_count"])
    if agreement["pairwise_kappa"]:
        st.dataframe(pd.DataFrame(agreement["pairwise_kappa"]), hide_index=True, width="stretch")
    else:
        st.info("Pairwise agreement appears after two reviewers label the same example.")
    st.caption("Kappa is descriptive; small samples and class imbalance can make it unstable.")
    with lock:
        votes = workbench.store.annotations()
        csv_export = workbench.store.export_csv()
    if votes:
        st.dataframe(pd.DataFrame(votes), hide_index=True, width="stretch")
    own_votes = {row["sample_id"]: row for row in votes if row["reviewer"] == reviewer}
    if own_votes:
        with st.expander("Revise one of your saved reviews"):
            revision_id = st.selectbox("Your saved review", list(own_votes))
            current_vote = own_votes[revision_id]
            revision_image = (workbench.dataset.X[revision_id].reshape(8, 8) * (255 / 16)).astype(
                np.uint8
            )
            st.image(np.repeat(np.repeat(revision_image, 16, axis=0), 16, axis=1), width=128)
            with st.form("revision"):
                revision_label = st.radio(
                    "Corrected digit",
                    list(range(10)),
                    index=current_vote["label"],
                    horizontal=True,
                    key=f"revision-label-{revision_id}-{reviewer}",
                )
                revision_note = st.text_area(
                    "Updated review note",
                    value=current_vote["note"],
                    max_chars=2000,
                    key=f"revision-note-{revision_id}-{reviewer}",
                )
                revise = st.form_submit_button("Save revision")
            if revise:
                with lock:
                    workbench.annotate(revision_id, reviewer, revision_label, revision_note)
                st.rerun()
    st.download_button("Export latest review votes", csv_export, "annotations.csv", mime="text/csv")
    st.caption(
        "Revisions append to the audit log. The table and export show each reviewer's latest vote."
    )
    with st.expander("Full annotation audit"):
        with lock:
            events = workbench.store.events()
        if events:
            st.dataframe(pd.DataFrame(events), hide_index=True, width="stretch")
        else:
            st.caption("The audit starts with the first saved review.")

with benchmark_tab:
    st.subheader("Does uncertainty sampling help?")
    st.markdown(
        "The reproducible experiment compares **entropy sampling** with **random sampling** "
        "at equal label budgets across three seeds. Acquisition uses reference labels as a "
        "simulated oracle; these are not human annotation results."
    )
    report_dir = Path(__file__).parent / "reports"
    curve_path = report_dir / "learning_curve.csv"
    if curve_path.exists():
        curves = pd.read_csv(curve_path)
        st.dataframe(curves, hide_index=True, width="stretch")
        st.caption(
            "Validation curves describe the fixed experiment. Final test scores are reported separately."
        )
        plot_path = report_dir / "learning_curve.svg"
        if plot_path.exists():
            st.image(str(plot_path), width="stretch")
        report_path = report_dir / "report.md"
        if report_path.exists():
            st.markdown(report_path.read_text())
        with st.expander("Machine-readable benchmark report"):
            json_path = report_dir / "report.json"
            if json_path.exists():
                st.json(json.loads(json_path.read_text()))
    else:
        st.code("labelops benchmark --output reports", language="bash")
        st.info("Run the benchmark to generate measured results and learning curves.")
    with st.expander("Evaluate this workspace's model on the sealed test split"):
        st.caption(
            "Use this after fixing the training procedure. Repeated checks are not independent evidence."
        )
        if st.button("Evaluate final model"):
            with lock:
                evaluation = workbench.evaluation()
            st.json(evaluation)
