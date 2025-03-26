# Streamlit TOPSIS Tool (Manual AHP Weights Input)
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="TOPSIS Renewable Energy Prioritization Tool", layout="wide")
st.title("⚖️ TOPSIS-Based Decision Support Tool for Renewable Energy")

# -------------------- USER INPUT --------------------
st.header("1️⃣ Define Criteria and Alternatives")

default_criteria = ["Cost", "Capacity Factor", "Resource Potential"]
criteria_input = st.text_area("Enter criteria (one per line):", "\n".join(default_criteria))
criteria = [c.strip() for c in criteria_input.split("\n") if c.strip()]

alternatives_input = st.text_area("Enter alternatives (one per line):", "Solar PV\nWind\nHydro")
alternatives = [alt.strip() for alt in alternatives_input.split("\n") if alt.strip()]

if len(criteria) < 2 or len(alternatives) < 2:
    st.warning("Please enter at least 2 criteria and 2 alternatives.")
    st.stop()

# -------------------- MANUAL WEIGHTS --------------------
st.header("2️⃣ Enter Weights from AHP Questionnaire")
st.write("Enter normalized weights (they should sum to 1.0)")

weights = []
total_weight = 0
for crit in criteria:
    w = st.number_input(f"Weight for {crit}", min_value=0.0, max_value=1.0, value=round(1/len(criteria), 2), step=0.01)
    weights.append(w)
    total_weight += w

if abs(total_weight - 1.0) > 0.01:
    st.warning(f"Weights should sum to 1.0. Current sum: {total_weight:.2f}")
    st.stop()

weights = np.array(weights)

# -------------------- TOPSIS DECISION MATRIX --------------------
st.header("3️⃣ Enter Performance Matrix")

performance_data = []
for alt in alternatives:
    row = []
    for crit in criteria:
        val = st.number_input(f"{alt} – {crit}", value=1.0, step=0.1, key=f"perf_{alt}_{crit}")
        row.append(val)
    performance_data.append(row)

decision_matrix = pd.DataFrame(performance_data, columns=criteria, index=alternatives)

# -------------------- CRITERIA TYPES --------------------
st.header("4️⃣ Define Criteria Type")
st.write("Specify whether each criterion is a Benefit (more is better) or a Cost (less is better)")

criterion_types = {}
for crit in criteria:
    ctype = st.radio(f"{crit} is a...", ["Benefit (more is better)", "Cost (less is better)"],
                     key=f"type_{crit}")
    criterion_types[crit] = (ctype == "Benefit (more is better)")

# -------------------- TOPSIS CALCULATION --------------------
st.header("5️⃣ TOPSIS Ranking")

# Normalize
norm = decision_matrix.copy()
for col in norm.columns:
    denom = np.sqrt((norm[col]**2).sum())
    norm[col] = norm[col] / denom if denom != 0 else 0

# Apply weights
weighted = norm * weights

# Determine ideal and anti-ideal
ideal = weighted.copy()
anti_ideal = weighted.copy()
for crit in criteria:
    if criterion_types[crit]:
        ideal[crit] = weighted[crit].max()
        anti_ideal[crit] = weighted[crit].min()
    else:
        ideal[crit] = weighted[crit].min()
        anti_ideal[crit] = weighted[crit].max()

# Calculate distances
d_pos = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
d_neg = np.sqrt(((weighted - anti_ideal) ** 2).sum(axis=1))
scores = d_neg / (d_pos + d_neg)
scores_clean = scores.fillna(0)
ranks = scores_clean.rank(ascending=False).fillna(len(scores_clean)).astype(int)

# Create result DataFrame
results = pd.DataFrame({
    "Alternative": alternatives,
    "TOPSIS Score": scores_clean.round(4),
    "Rank": ranks
}).sort_values("TOPSIS Score", ascending=False).reset_index(drop=True)

# -------------------- DISPLAY RESULTS --------------------
st.subheader("✅ Final TOPSIS Ranking")
st.dataframe(results.style.highlight_max(subset=['TOPSIS Score'], color='lightgreen'), use_container_width=True)

# Bar chart
fig = px.bar(results, x="Alternative", y="TOPSIS Score", color="TOPSIS Score",
             color_continuous_scale="Aggrnyl", text_auto=True)
fig.update_layout(yaxis_title="Score", xaxis_title="Alternative")
st.plotly_chart(fig, use_container_width=True)

st.success("🎯 TOPSIS prioritization complete. You can now use this output for decision-making.")