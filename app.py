
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Page config
st.set_page_config(page_title="Renewable Energy Decision Tool", layout="centered")

# Google Fonts + Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif;
        background: #f4f7f9;
        color: #1c1c1e;
    }
    .stApp {
        padding: 2rem;
    }
    .step-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        max-width: 600px;
        margin: auto;
        box-shadow: 0 12px 24px rgba(0,0,0,0.05);
    }
    .stButton>button {
        padding: 0.75rem 2rem;
        font-size: 1rem;
        border-radius: 12px;
        background-color: #2563eb;
        color: white;
        border: none;
        margin: 0.5rem 1rem 0.5rem 0;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
    }
    .stRadio label {
        padding: 0.6rem 1rem;
        border-radius: 12px;
        background-color: #f0f0f5;
        margin-bottom: 0.5rem;
        display: block;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize state
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

criteria = ["Cost", "Efficiency", "Environmental Impact", "Social Acceptance"]
alternatives = ["Solar", "Wind", "Hydro"]
feedback_questions = [
    "The tool is easy to use.",
    "I understand how my inputs affect the result.",
    "The app is useful for decision-making.",
    "I would recommend this tool to others."
]
total_steps = 1 + 1 + len(alternatives)*len(criteria) + len(feedback_questions) + 1
st.progress((st.session_state.step + 1) / total_steps)

def card(content_func):
    with st.container():
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        content_func()
        st.markdown('</div>', unsafe_allow_html=True)

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

# --- Pages ---
if st.session_state.step == 0:
    def content():
        st.markdown("### Welcome to the Renewable Energy Decision Support Tool")
        st.write("This tool helps prioritize renewable energy options in Indonesia using AHP and TOPSIS.")
        st.markdown("""
        - ✅ Fixed AHP weights (for research consistency)
        - ✅ Rate energy alternatives
        - ✅ View real-time aggregated ranking
        - ✅ Share your feedback
        """)
    card(content)
    nav_buttons()

elif st.session_state.step == 1:
    def content():
        st.markdown("### AHP Criteria Weights")
        st.write("Below are the fixed weights. You can enable editing for demo or policymaker use.")
        st.toggle("Enable weight editing", key="editable_weights")
        total = 0
        new_weights = []
        for i, crit in enumerate(criteria):
            w = st.slider(crit, 0.0, 1.0, st.session_state.weights[i], 0.01, key=f"w_{i}", disabled=not st.session_state.editable_weights)
            new_weights.append(w)
            total += w
        st.write(f"**Total Weight:** {round(total, 2)} (must equal 1.0)")
        if not st.session_state.editable_weights or round(total, 2) == 1.0:
            st.session_state.weights = new_weights
    card(content)
    nav_buttons()

elif st.session_state.step < 2 + len(alternatives)*len(criteria):
    idx = st.session_state.step - 2
    alt = alternatives[idx // len(criteria)]
    crit = criteria[idx % len(criteria)]
    def content():
        st.markdown(f"### Rate: {alt} on {crit}")
        st.write("Select how well this alternative performs.")
        st.session_state.scores[f"{alt}_{crit}"] = st.radio("Your rating", [1, 2, 3, 4, 5], horizontal=True, key=f"{alt}_{crit}")
    card(content)
    nav_buttons()

elif st.session_state.step < 2 + len(alternatives)*len(criteria) + len(feedback_questions):
    fb_idx = st.session_state.step - (2 + len(alternatives)*len(criteria))
    q = feedback_questions[fb_idx]
    def content():
        st.markdown(f"### Feedback")
        st.write(q)
        st.session_state.feedback[q] = st.radio("Your rating", [1, 2, 3, 4, 5], horizontal=True, key=f"fb_{fb_idx}")
    card(content)
    nav_buttons()

elif st.session_state.step == total_steps - 1:
    def content():
        st.markdown("### Final Comments")
        st.write("Feel free to leave suggestions or feedback.")
        st.session_state.comment = st.text_area("Your comment", value=st.session_state.comment)
    card(content)
    if st.button("✅ Submit"):
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
        st.success("🎉 Thank you for your input!")
        st.session_state.step += 1

else:
    st.markdown("## Aggregated TOPSIS Results")
    if os.path.exists("user_data.csv"):
        df_all = pd.read_csv("user_data.csv")
        weights_avg = [df_all[f"weight_{c}"].astype(float).mean() for c in criteria]
        matrix = []
        for alt in alternatives:
            matrix.append([df_all[f"{alt}_{c}"].astype(float).mean() for c in criteria])
        matrix = np.array(matrix)
        norm_matrix = matrix / np.sqrt((matrix**2).sum(axis=0))
        weighted_matrix = norm_matrix * np.array(weights_avg)
        ideal_best = weighted_matrix.max(axis=0)
        ideal_worst = weighted_matrix.min(axis=0)
        dist_best = np.linalg.norm(weighted_matrix - ideal_best, axis=1)
        dist_worst = np.linalg.norm(weighted_matrix - ideal_worst, axis=1)
        scores = dist_worst / (dist_best + dist_worst)
        topsis_df = pd.DataFrame({"Alternative": alternatives, "TOPSIS Score": scores}).sort_values(by="TOPSIS Score", ascending=False)
        st.dataframe(topsis_df, use_container_width=True)
        st.download_button("⬇️ Download All Data", df_all.to_csv(index=False), "user_data.csv")
        st.download_button("⬇️ Download TOPSIS Ranking", topsis_df.to_csv(index=False), "topsis.csv")
    else:
        st.info("No data submitted yet.")
