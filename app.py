
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

st.set_page_config(page_title="AHP-TOPSIS Survey", layout="centered")

# Session state to track steps
if "step" not in st.session_state:
    st.session_state.step = 0
if "responses" not in st.session_state:
    st.session_state.responses = {}
if "scores" not in st.session_state:
    st.session_state.scores = {}
if "comment" not in st.session_state:
    st.session_state.comment = ""

# Static data
criteria = ["Cost", "Efficiency", "Environmental Impact", "Social Acceptance"]
weights = np.array([0.3, 0.25, 0.25, 0.2])
alternatives = ["Solar", "Wind", "Hydro"]
feedback_questions = [
    "The tool is easy to use.",
    "I understand how my inputs affect the result.",
    "The app is useful for decision-making.",
    "I would recommend this tool to others."
]

total_steps = len(alternatives) * len(criteria) + len(feedback_questions) + 1

# Progress
st.progress((st.session_state.step + 1) / total_steps)

# STEP HANDLER
current = st.session_state.step

# --- STEP: Rate Alternatives per Criterion ---
alt_idx = current // len(criteria)
crit_idx = current % len(criteria)

if current < len(alternatives) * len(criteria):
    alt = alternatives[alt_idx]
    crit = criteria[crit_idx]
    st.header(f"How do you rate **{alt}** for **{crit}**?")
    choice = st.radio("Select a score (1 = Poor, 5 = Excellent)", [1, 2, 3, 4, 5], horizontal=True)
    if st.button("Next"):
        key = f"{alt}_{crit}"
        st.session_state.scores[key] = choice
        st.session_state.step += 1

# --- STEP: Feedback Questions ---
elif current < len(alternatives) * len(criteria) + len(feedback_questions):
    fb_idx = current - len(alternatives) * len(criteria)
    q = feedback_questions[fb_idx]
    st.header(q)
    choice = st.radio("Your response", [1, 2, 3, 4, 5], horizontal=True)
    if st.button("Next"):
        st.session_state.responses[q] = choice
        st.session_state.step += 1

# --- STEP: Comment + Submit ---
elif current == total_steps - 1:
    st.header("Any final comments or suggestions?")
    st.session_state.comment = st.text_area("Your feedback here...")
    if st.button("Submit"):
        # Save input
        row = {
            "timestamp": datetime.now().isoformat(),
            "comment": st.session_state.comment
        }
        for q in feedback_questions:
            row[q] = st.session_state.responses.get(q, "")
        for alt in alternatives:
            for crit in criteria:
                key = f"{alt}_{crit}"
                row[key] = st.session_state.scores.get(key, "")
        df = pd.DataFrame([row])
        file_exists = os.path.exists("user_data.csv")
        df.to_csv("user_data.csv", mode='a', header=not file_exists, index=False)
        st.success("✅ Thank you! Your feedback has been recorded.")
        st.session_state.step += 1

# --- STEP: Thank You & Aggregated Results ---
else:
    st.header("🎉 Thank you for participating!")
    st.write("Here's the current aggregated result from all responses:")

    if os.path.exists("user_data.csv"):
        df_all = pd.read_csv("user_data.csv")
        matrix = []
        for alt in alternatives:
            alt_scores = []
            for crit in criteria:
                alt_scores.append(df_all[f"{alt}_{crit}"].astype(float).mean())
            matrix.append(alt_scores)

        matrix = np.array(matrix)
        norm_matrix = matrix / np.sqrt((matrix**2).sum(axis=0))
        weighted_matrix = norm_matrix * weights

        ideal_best = weighted_matrix.max(axis=0)
        ideal_worst = weighted_matrix.min(axis=0)

        distances_best = np.linalg.norm(weighted_matrix - ideal_best, axis=1)
        distances_worst = np.linalg.norm(weighted_matrix - ideal_worst, axis=1)

        scores = distances_worst / (distances_best + distances_worst)
        topsis_df = pd.DataFrame({
            "Alternative": alternatives,
            "TOPSIS Score": scores
        }).sort_values(by="TOPSIS Score", ascending=False).reset_index(drop=True)

        st.table(topsis_df)
    else:
        st.info("No results yet.")
