
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

st.set_page_config(page_title="AHP-TOPSIS Tool", layout="centered")

# Session State Initialization
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
if "editable_weights" not in st.session_state:
    st.session_state.editable_weights = False

# Static Data
criteria = ["Cost", "Efficiency", "Environmental Impact", "Social Acceptance"]
alternatives = ["Solar", "Wind", "Hydro"]
feedback_questions = [
    "The tool is easy to use.",
    "I understand how my inputs affect the result.",
    "The app is useful for decision-making.",
    "I would recommend this tool to others."
]

# Total Steps
total_steps = 1 + 1 + len(alternatives)*len(criteria) + len(feedback_questions) + 1

# Progress Bar
st.progress((st.session_state.step + 1) / total_steps)

# Navigation Buttons
def nav_buttons():
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.step > 0:
            if st.button("⬅️ Previous"):
                st.session_state.step -= 1
    with col2:
        if st.session_state.step < total_steps - 1:
            if st.button("Next ➡️"):
                st.session_state.step += 1

# --- Step 0: Welcome ---
if st.session_state.step == 0:
    st.title("🔋 Renewable Energy Decision Support Tool")
    st.markdown("""
    Welcome to this AHP-TOPSIS-based survey tool designed to support renewable energy decision-making in Indonesia.

    💡 In this tool:
    - You will review predefined AHP weights (or adjust them if enabled)
    - Rate renewable alternatives (Solar, Wind, Hydro)
    - See aggregated decision results
    - Provide feedback about your experience

    Click **Next** to begin.
    """)
    nav_buttons()

# --- Step 1: AHP Weights ---
elif st.session_state.step == 1:
    st.title("⚖️ AHP Criteria Weights")
    st.markdown("These weights determine the importance of each criterion in the decision-making process.")
    st.toggle("Enable weight adjustment (for demonstration)", key="editable_weights")

    new_weights = []
    total = 0
    for i, crit in enumerate(criteria):
        default_val = st.session_state.weights[i]
        disabled = not st.session_state.editable_weights
        weight = st.slider(crit, 0.0, 1.0, default_val, 0.01, key=f"weight_{i}", disabled=disabled)
        new_weights.append(weight)
        total += weight
    st.write(f"**Total Weight:** {round(total, 2)} (should be 1.0)")

    if not st.session_state.editable_weights or round(total, 2) == 1.0:
        st.session_state.weights = new_weights
        nav_buttons()
    else:
        st.warning("Total weight must equal 1.0 to proceed.")

# --- Step 2+: Rating Alternatives ---
elif st.session_state.step < 2 + len(alternatives)*len(criteria):
    idx = st.session_state.step - 2
    alt = alternatives[idx // len(criteria)]
    crit = criteria[idx % len(criteria)]
    st.header(f"📌 Rate: **{alt}** for **{crit}**")
    st.markdown("Rate how well this energy source performs under the given criterion.")
    choice = st.radio("Select a score (1 = Poor, 5 = Excellent)", [1, 2, 3, 4, 5],
                      horizontal=True, key=f"{alt}_{crit}")
    st.session_state.scores[f"{alt}_{crit}"] = choice
    nav_buttons()

# --- Feedback Pages ---
elif st.session_state.step < 2 + len(alternatives)*len(criteria) + len(feedback_questions):
    fb_idx = st.session_state.step - (2 + len(alternatives)*len(criteria))
    q = feedback_questions[fb_idx]
    st.header(f"💬 Feedback: {q}")
    st.session_state.feedback[q] = st.radio("Your rating", [1, 2, 3, 4, 5], horizontal=True, key=f"fb_{fb_idx}")
    nav_buttons()

# --- Final Comments + Submit ---
elif st.session_state.step == total_steps - 1:
    st.header("📝 Final Comments")
    st.session_state.comment = st.text_area("Anything you'd like to share or suggest?", value=st.session_state.comment)
    if st.button("✅ Submit All"):
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
        st.success("🎉 Submission recorded! Proceed to view the results.")
        st.session_state.step += 1

# --- Show Aggregated Results ---
else:
    st.title("📊 Aggregated TOPSIS Ranking")
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
        st.download_button("⬇️ Download User Data", df_all.to_csv(index=False), "user_data.csv", "text/csv")
        st.download_button("⬇️ Download TOPSIS Ranking", topsis_df.to_csv(index=False), "topsis_ranking.csv", "text/csv")
    else:
        st.info("No results available yet. Please submit responses to populate the ranking.")
