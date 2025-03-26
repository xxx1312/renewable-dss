
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

# ----------------------------
# Streamlit Config & Styling
# ----------------------------
st.set_page_config(page_title="Renewable Energy AHP-TOPSIS", layout="centered")

def styled_card(content):
    st.markdown(
        f"""
        <div style="background-color: #ffffffdd; padding: 2rem; border-radius: 20px; box-shadow: 0 0 20px rgba(0,0,0,0.05); margin: auto; max-width: 600px;">
            {content}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <style>
    .stButton>button {
        padding: 0.75rem 2rem;
        font-size: 1rem;
        border-radius: 12px;
        margin: 0.5rem 1rem;
    }
    .stRadio label {
        padding: 0.75rem 1rem;
        border-radius: 12px;
        background-color: #f2f2f2;
        margin: 0.3rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# App State Initialization
# ----------------------------
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

# ----------------------------
# App Pages
# ----------------------------
if st.session_state.step == 0:
    styled_card("""
    <h2>🔋 Renewable Energy Decision Tool</h2>
    <p>This interactive AHP-TOPSIS tool supports prioritization of renewable energy alternatives in Indonesia.</p>
    <ul>
        <li>Review or adjust AHP weights (criteria importance)</li>
        <li>Rate alternatives on multiple criteria</li>
        <li>Get a ranked result and provide your feedback</li>
    </ul>
    """)
    nav_buttons()

elif st.session_state.step == 1:
    content = "<h3>⚖️ Adjust AHP Criteria Weights</h3>"
    content += "<p>You can optionally adjust the importance of each decision criterion. Total must equal 1.0.</p>"
    styled_card(content)
    st.toggle("Enable weight editing", key="editable_weights")
    new_weights = []
    total = 0
    for i, crit in enumerate(criteria):
        w = st.slider(crit, 0.0, 1.0, st.session_state.weights[i], 0.01, key=f"w_{i}", disabled=not st.session_state.editable_weights)
        new_weights.append(w)
        total += w
    st.write(f"**Total:** {round(total, 2)}")
    if not st.session_state.editable_weights or round(total, 2) == 1.0:
        st.session_state.weights = new_weights
        nav_buttons()
    else:
        st.warning("Weights must sum to 1.0")

elif st.session_state.step < 2 + len(alternatives)*len(criteria):
    idx = st.session_state.step - 2
    alt = alternatives[idx // len(criteria)]
    crit = criteria[idx % len(criteria)]
    styled_card(f"""
    <h3>📌 Rate: <strong>{alt}</strong> on <strong>{crit}</strong></h3>
    <p>Select how well this alternative performs on this criterion.</p>
    """)
    st.session_state.scores[f"{alt}_{crit}"] = st.radio("Your score (1 = Poor, 5 = Excellent)", [1, 2, 3, 4, 5], horizontal=True, key=f"{alt}_{crit}")
    nav_buttons()

elif st.session_state.step < 2 + len(alternatives)*len(criteria) + len(feedback_questions):
    fb_idx = st.session_state.step - (2 + len(alternatives)*len(criteria))
    q = feedback_questions[fb_idx]
    styled_card(f"<h3>💬 Feedback</h3><p>{q}</p>")
    st.session_state.feedback[q] = st.radio("Your rating", [1, 2, 3, 4, 5], horizontal=True, key=f"fb_{fb_idx}")
    nav_buttons()

elif st.session_state.step == total_steps - 1:
    styled_card("<h3>📝 Final Comments</h3><p>Anything else you’d like to share?</p>")
    st.session_state.comment = st.text_area("Your comment", value=st.session_state.comment)
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
        st.success("🎉 Submission received!")
        st.session_state.step += 1

else:
    st.title("📊 Aggregated TOPSIS Results")
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
        st.download_button("⬇️ Download Data", df_all.to_csv(index=False), "user_data.csv", "text/csv")
        st.download_button("⬇️ Download Ranking", topsis_df.to_csv(index=False), "topsis_ranking.csv", "text/csv")
    else:
        st.info("No submissions yet.")
