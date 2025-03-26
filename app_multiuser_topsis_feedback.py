# Streamlit Multi-User TOPSIS Tool with Fixed AHP Weights and Feedback Survey
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import uuid

st.set_page_config(page_title="Multi-User TOPSIS + Feedback Tool", layout="wide")
st.title("⚖️ Renewable Energy Prioritization (TOPSIS) Tool")

if "user_results" not in st.session_state:
    st.session_state.user_results = []

if "feedback" not in st.session_state:
    st.session_state.feedback = []

st.header("1️⃣ Set Criteria, Alternatives, and AHP Weights")
criteria_input = st.text_area("Enter criteria (one per line):", "Cost\nCapacity Factor\nResource Potential")
alternatives_input = st.text_area("Enter alternatives (one per line):", "Solar PV\nWind\nHydro")

criteria = [c.strip() for c in criteria_input.split("\n") if c.strip()]
alternatives = [a.strip() for a in alternatives_input.split("\n") if a.strip()]

weights = []
total_weight = 0
st.subheader("AHP Weights (Sum must be 1.0)")
for crit in criteria:
    w = st.number_input(f"Weight for {crit}", min_value=0.0, max_value=1.0, value=round(1/len(criteria), 2), step=0.01)
    weights.append(w)
    total_weight += w

if abs(total_weight - 1.0) > 0.01:
    st.warning(f"Weights should sum to 1.0. Current sum: {total_weight:.2f}")
    st.stop()

weights = np.array(weights)

st.header("2️⃣ Enter Performance Matrix")
performance_data = []
for alt in alternatives:
    row = []
    for crit in criteria:
        val = st.number_input(f"{alt} – {crit}", value=1.0, step=0.1, key=f"input_{alt}_{crit}")
        row.append(val)
    performance_data.append(row)

decision_matrix = pd.DataFrame(performance_data, columns=criteria, index=alternatives)

st.subheader("Criteria Type")
criterion_types = {}
for crit in criteria:
    ctype = st.radio(f"{crit} is a...", ["Benefit", "Cost"], key=f"type_{crit}")
    criterion_types[crit] = (ctype == "Benefit")

if st.button("Submit and Calculate TOPSIS"):
    norm = decision_matrix.copy()
    for col in norm.columns:
        denom = np.sqrt((norm[col]**2).sum())
        norm[col] = norm[col] / denom if denom != 0 else 0

    weighted = norm * weights

    ideal, anti_ideal = weighted.copy(), weighted.copy()
    for crit in criteria:
        if criterion_types[crit]:
            ideal[crit] = weighted[crit].max()
            anti_ideal[crit] = weighted[crit].min()
        else:
            ideal[crit] = weighted[crit].min()
            anti_ideal[crit] = weighted[crit].max()

    d_pos = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    d_neg = np.sqrt(((weighted - anti_ideal) ** 2).sum(axis=1))
    scores = d_neg / (d_pos + d_neg)
    scores_clean = scores.fillna(0)
    ranks = scores_clean.rank(ascending=False).fillna(len(scores_clean)).astype(int)

    user_result = pd.DataFrame({
        "Alternative": alternatives,
        "TOPSIS Score": scores_clean.round(4),
        "Rank": ranks
    }).sort_values("TOPSIS Score", ascending=False).reset_index(drop=True)

    st.session_state.user_results.append(user_result)

    st.subheader("Your Ranking")
    st.dataframe(user_result, use_container_width=True)
    fig = px.bar(user_result, x="Alternative", y="TOPSIS Score", color="TOPSIS Score",
                 color_continuous_scale="Aggrnyl", text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

    st.success("✅ Your response has been recorded.")

    with st.expander("Step 3️⃣: Quick Feedback Survey"):
        q1 = st.slider("This tool was easy to use", 1, 5, 3)
        q2 = st.slider("The results were easy to understand", 1, 5, 3)
        q3 = st.slider("This tool supports better decision-making", 1, 5, 3)
        q4 = st.slider("I would recommend this tool to others", 1, 5, 3)

        if st.button("Submit Feedback"):
            st.session_state.feedback.append({"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4})
            st.success("🎉 Thank you for your feedback!")

st.header("📊 Aggregated Results")
if st.session_state.user_results:
    all_scores = [df.set_index("Alternative")["TOPSIS Score"] for df in st.session_state.user_results]
    avg_scores = pd.concat(all_scores, axis=1).mean(axis=1)
    avg_ranks = avg_scores.rank(ascending=False).astype(int)
    summary = pd.DataFrame({"Average Score": avg_scores.round(4), "Rank": avg_ranks}).sort_values("Average Score", ascending=False)
    st.dataframe(summary, use_container_width=True)

st.header("📝 Feedback Summary")
if st.session_state.feedback:
    df_feedback = pd.DataFrame(st.session_state.feedback)
    st.dataframe(df_feedback.describe().T, use_container_width=True)
