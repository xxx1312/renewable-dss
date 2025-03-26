
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

st.set_page_config(page_title="AHP-TOPSIS Multi-user Decision Tool", layout="centered")

st.title("🔋 Renewable Energy Decision Support Tool")
st.write("This tool helps prioritize renewable energy alternatives using AHP-TOPSIS. Your input contributes to an aggregated decision ranking.")

# --- AHP Weights (fixed) ---
criteria = ["Cost", "Efficiency", "Environmental Impact", "Social Acceptance"]
weights = np.array([0.3, 0.25, 0.25, 0.2])
st.subheader("📊 AHP Weights (Predefined)")
for c, w in zip(criteria, weights):
    st.write(f"- **{c}**: {w}")

# --- User TOPSIS Input ---
st.subheader("📝 Rate Each Alternative (1 = Poor, 5 = Excellent)")

alternatives = ["Solar", "Wind", "Hydro"]
user_scores = {}

for alt in alternatives:
    user_scores[alt] = []
    st.markdown(f"**{alt}**")
    cols = st.columns(len(criteria))
    for i, crit in enumerate(criteria):
        with cols[i]:
            score = st.selectbox(
                f"{crit} ({alt})",
                [1, 2, 3, 4, 5],
                key=f"{alt}_{crit}"
            )
            user_scores[alt].append(score)

# --- Likert Scale Feedback ---
st.subheader("💬 App Feedback (1 = Strongly Disagree, 5 = Strongly Agree)")

feedback_questions = [
    "The tool is easy to use.",
    "I understand how my inputs affect the result.",
    "The app is useful for decision-making.",
    "I would recommend this tool to others."
]

feedback_responses = {}
for q in feedback_questions:
    feedback_responses[q] = st.slider(q, 1, 5, 3)

comments = st.text_area("Additional comments or suggestions")

if st.button("Submit"):
    # Save input
    result = {
        "timestamp": datetime.now().isoformat(),
        "comments": comments
    }
    for q in feedback_questions:
        result[q] = feedback_responses[q]
    for alt in alternatives:
        for i, crit in enumerate(criteria):
            result[f"{alt}_{crit}"] = user_scores[alt][i]

    df = pd.DataFrame([result])
    file_exists = os.path.exists("user_data.csv")
    df.to_csv("user_data.csv", mode='a', header=not file_exists, index=False)
    st.success("✅ Your input has been submitted. Thank you!")

# --- Aggregated TOPSIS Calculation ---
st.subheader("📈 Aggregated TOPSIS Ranking")

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
    st.info("No user input yet. Be the first to submit!")
