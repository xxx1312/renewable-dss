# Streamlit AHP + TOPSIS Decision Support Tool for Renewable Energy
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Renewable Energy Prioritization Tool", layout="wide")
st.title("🌿 AHP + TOPSIS Decision Support Tool for Renewable Energy")

# -------------------- USER INPUT --------------------
st.header("1️⃣ Define Criteria and Alternatives")

criteria = st.multiselect("Enter Criteria (e.g., Cost, Emissions, Potential):",
                          options=[],
                          default=["Cost", "Capacity Factor", "Resource Potential"])

alternatives = st.text_area("Enter Alternatives (one per line):", "Solar PV\nWind\nHydro")
alternatives = [alt.strip() for alt in alternatives.split("\n") if alt.strip()]

if len(criteria) < 2 or len(alternatives) < 2:
    st.warning("Please enter at least 2 criteria and 2 alternatives.")
    st.stop()

# -------------------- AHP PAIRWISE INPUT --------------------
st.header("2️⃣ Pairwise Comparison Matrix (AHP)")
st.write("Provide judgments to compare the importance of criteria (1–9 scale). Reciprocal values are auto-filled.")

n = len(criteria)
pairwise_matrix = np.ones((n, n))

ahp_table = pd.DataFrame(pairwise_matrix, columns=criteria, index=criteria)

for i in range(n):
    for j in range(i+1, n):
        val = st.number_input(f"How much more important is '{criteria[i]}' over '{criteria[j]}'? (1–9)",
                              min_value=1.0, max_value=9.0, value=1.0, step=0.1,
                              key=f"pc_{i}_{j}")
        ahp_table.iloc[i, j] = val
        ahp_table.iloc[j, i] = 1 / val

# AHP Calculations
norm_matrix = ahp_table / ahp_table.sum(axis=0)
weights = norm_matrix.mean(axis=1)
weighted_sum = ahp_table @ weights
lambda_max = (weighted_sum / weights).mean()
ci = (lambda_max - n) / (n - 1)
ri_dict = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
ri = ri_dict.get(n, 1.49)
cr = ci / ri if ri else 0

st.subheader("Criteria Weights (from AHP)")
st.dataframe(weights.round(4).to_frame(name="Weight"))
st.markdown(f"**Consistency Ratio (CR):** {cr:.4f} " + ("✅ Acceptable" if cr < 0.1 else "⚠️ Review your inputs"))

# -------------------- TOPSIS MATRIX --------------------
st.header("3️⃣ Enter Performance Matrix (TOPSIS)")
st.write("Fill in the performance of each alternative under each criterion.")

performance_data = []
for alt in alternatives:
    row = []
    for crit in criteria:
        val = st.number_input(f"{alt} – {crit}", value=1.0, step=0.1,
                              key=f"perf_{alt}_{crit}")
        row.append(val)
    performance_data.append(row)

decision_matrix = pd.DataFrame(performance_data, columns=criteria, index=alternatives)

# Determine Benefit/Cost Criteria
st.subheader("Define Criteria Type")
criterion_types = {}
for crit in criteria:
    ctype = st.radio(f"{crit} is a...", ["Benefit (more is better)", "Cost (less is better)"],
                     key=f"type_{crit}")
    criterion_types[crit] = (ctype == "Benefit (more is better)")

# -------------------- TOPSIS CALCULATION --------------------
st.header("4️⃣ TOPSIS Ranking")

# Normalize
norm = decision_matrix / np.sqrt((decision_matrix**2).sum())
# Weighted
weighted = norm * weights.values

# Ideal & Anti-Ideal
ideal = weighted.copy()
anti_ideal = weighted.copy()
for crit in criteria:
    if criterion_types[crit]:  # benefit
        ideal[crit] = weighted[crit].max()
        anti_ideal[crit] = weighted[crit].min()
    else:  # cost
        ideal[crit] = weighted[crit].min()
        anti_ideal[crit] = weighted[crit].max()

# Distances
d_pos = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
d_neg = np.sqrt(((weighted - anti_ideal) ** 2).sum(axis=1))
scores = d_neg / (d_pos + d_neg)

results = pd.DataFrame({
    "Alternative": alternatives,
    "TOPSIS Score": scores,
    "Rank": scores.rank(ascending=False).astype(int)
}).sort_values("TOPSIS Score", ascending=False).reset_index(drop=True)

st.subheader("Final Ranking")
st.dataframe(results)

fig = px.bar(results, x="Alternative", y="TOPSIS Score", color="TOPSIS Score",
             color_continuous_scale="Aggrnyl", text_auto=True)
fig.update_layout(yaxis_title="Score", xaxis_title="Alternative")
st.plotly_chart(fig, use_container_width=True)

st.success("🎉 Done! You can now export this or deploy it on Streamlit Cloud.")