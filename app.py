
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

st.set_page_config(page_title="AHP-TOPSIS Decision Tool", layout="centered")

# --- Session State ---
if "step" not in st.session_state:
    st.session_state.step = 0
if "weights" not in st.session_state:
    st.session_state.weights = [0.3, 0.25, 0.25, 0.2]
if "scores" not in st.session_state:
    st.session_state.scores = {}
if "feedback" not in st.session_state:
    st.session_state.feedback = {}
if "comment" not in st.session_state:
    st.session_state.comment = ""

# --- Data Setup ---
criteria = ["Cost", "Efficiency", "Environmental Impact", "Social Acceptance"]
alternatives = ["Solar", "Wind", "Hydro"]
feedback_questions = [
    "The tool is easy to use.",
    "I understand how my inputs affect the result.",
    "The app is useful for decision-making.",
    "I would recommend this tool to others."
]

# --- Page Flow ---
total_steps = 1 + 1 + len(alternatives)*len(criteria) + len(feedback_questions) + 1

# --- Page 0: Intro ---
if st.session_state.step == 0:
    st.title("🔋 Renewable Energy Decision Support Tool")
    st.markdown("""
    Welcome to the AHP-TOPSIS-based decision support tool designed to assist policymakers 
    in prioritizing renewable energy alternatives.

    This interactive tool allows you to:
    - Adjust AHP weights for decision criteria
    - Rate each energy alternative
    - View an aggregated ranking based on TOPSIS
    - Provide feedback to improve the tool

    Click **Next** to begin.
    """)
    if st.button("Next"):
        st.session_state.step += 1

# --- Page 1: AHP Weight Adjustment ---
elif st.session_state.step == 1:
    st.title("⚖️ Adjust AHP Weights")
    st.markdown("Use the sliders to adjust importance of each criterion (total must be 1.0).")
    new_weights = []
    total = 0
    for i, crit in enumerate(criteria):
        weight = st.slider(crit, 0.0, 1.0, st.session_state.weights[i], 0.01, key=f"weight_{i}")
        new_weights.append(weight)
        total += weight
    st.write(f"**Total:** {round(total, 2)} (must equal 1.0)")
    if total == 1.0 and st.button("Next"):
        st.session_state.weights = new_weights
        st.session_state.step += 1

# --- Page 2–N: Rate Alternatives ---
elif st.session_state.step < 2 + len(alternatives)*len(criteria):
    idx = st.session_state.step - 2
    alt = alternatives[idx // len(criteria)]
    crit = criteria[idx % len(criteria)]
    st.header(f"Rate **{alt}** for **{crit}**")
    score = st.radio("Select a score (1 = Poor, 5 = Excellent)", [1, 2, 3, 4, 5], horizontal=True)
    if st.button("Next"):
        key = f"{alt}_{crit}"
        st.session_state.scores[key] = score
        st.session_state.step += 1

# --- Feedback Pages ---
elif st.session_state.step < 2 + len(alternatives)*len(criteria) + len(feedback_questions):
    fb_idx = st.session_state.step - (2 + len(alternatives)*len(criteria))
    q = feedback_questions[fb_idx]
    st.header(q)
    rating = st.radio("Your response", [1, 2, 3, 4, 5], horizontal=True)
    if st.button("Next"):
        st.session_state.feedback[q] = rating
        st.session_state.step += 1

# --- Comment + Submit ---
elif st.session_state.step == total_steps - 1:
    st.header("💬 Final Comments")
    st.session_state.comment = st.text_area("Anything you'd like to share?")
    if st.button("Submit"):
        row = {
            "timestamp": datetime.now().isoformat(),
            "comment": st.session_state.comment
        }
        for alt in alternatives:
            for crit in criteria:
                row[f"{alt}_{crit}"] = st.session_state.scores.get(f"{alt}_{crit}", "")
        for i, w in enumerate(st.session_state.weights):
            row[f"weight_{criteria[i]}"] = w
        for q in feedback_questions:
            row[q] = st.session_state.feedback.get(q, "")
        df = pd.DataFrame([row])
        file_exists = os.path.exists("user_data.csv")
        df.to_csv("user_data.csv", mode='a', header=not file_exists, index=False)
        st.success("✅ Submission recorded!")
        st.session_state.step += 1

# --- Final Page: Aggregated Results ---
else:
    st.title("📈 Aggregated TOPSIS Results")
    if os.path.exists("user_data.csv"):
        df_all = pd.read_csv("user_data.csv")
        weights_avg = []
        for crit in criteria:
            weights_avg.append(df_all[f"weight_{crit}"].astype(float).mean())
        matrix = []
        for alt in alternatives:
            row = []
            for crit in criteria:
                row.append(df_all[f"{alt}_{crit}"].astype(float).mean())
            matrix.append(row)
        matrix = np.array(matrix)
        norm_matrix = matrix / np.sqrt((matrix**2).sum(axis=0))
        weighted_matrix = norm_matrix * np.array(weights_avg)
        ideal_best = weighted_matrix.max(axis=0)
        ideal_worst = weighted_matrix.min(axis=0)
        dist_best = np.linalg.norm(weighted_matrix - ideal_best, axis=1)
        dist_worst = np.linalg.norm(weighted_matrix - ideal_worst, axis=1)
        scores = dist_worst / (dist_best + dist_worst)
        topsis_df = pd.DataFrame({
            "Alternative": alternatives,
            "TOPSIS Score": scores
        }).sort_values(by="TOPSIS Score", ascending=False).reset_index(drop=True)
        st.dataframe(topsis_df, use_container_width=True)
        st.download_button("⬇️ Download Raw Data", df_all.to_csv(index=False), "user_data.csv", "text/csv")
    else:
        st.info("No data yet. Be the first to submit.")
