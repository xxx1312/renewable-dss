# Streamlit Multi-User AHP + TOPSIS Decision Support Tool for Renewable Energy
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Multi-User Renewable DSS", layout="wide")
st.title("👥 Multi-User AHP + TOPSIS Decision Support Tool")

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

# -------------------- MULTI-USER AHP --------------------
st.header("2️⃣ User AHP Judgments")

user_id = st.text_input("Enter your name or ID:", key="user_id")

if not user_id:
    st.warning("Please enter your name or ID.")
    st.stop()

if "user_judgments" not in st.session_state:
    st.session_state.user_judgments = {}

n = len(criteria)
pairwise_matrix = np.ones((n, n))
ahp_table = pd.DataFrame(pairwise_matrix, columns=criteria, index=criteria)

for i in range(n):
    for j in range(i+1, n):
        val = st.number_input(f"[{user_id}] How much more important is '{criteria[i]}' over '{criteria[j]}'? (1–9)",
                              min_value=1.0, max_value=9.0, value=1.0, step=0.1,
                              key=f"pc_{user_id}_{i}_{j}")
        ahp_table.iloc[i, j] = val
        ahp_table.iloc[j, i] = 1 / val

if st.button("Submit Judgments"):
    st.session_state.user_judgments[user_id] = ahp_table.copy()
    st.success(f"Judgments submitted for {user_id}")

if len(st.session_state.user_judgments) < 1:
    st.stop()

# -------------------- AHP AGGREGATION --------------------
st.header("3️⃣ Aggregated AHP Weights")

all_matrices = list(st.session_state.user_judgments.values())
agg_matrix = np.ones((n, n))
for i in range(n):
    for j in range(n):
        product = np.prod([mat.iloc[i, j] for mat in all_matrices])
        agg_matrix[i, j] = product ** (1 / len(all_matrices))

agg_df = pd.DataFrame(agg_matrix, columns=criteria, index=criteria)
norm_matrix = agg_df / agg_df.sum(axis=0)
weights = norm_matrix.mean(axis=1)
weighted_sum = agg_df @ weights
lambda_max = (weighted_sum / weights).mean()
ci = (lambda_max - n) / (n - 1)
ri_dict = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
ri = ri_dict.get(n, 1.49)
cr = ci / ri if ri else 0

st.subheader("Aggregated Weights")
st.dataframe(weights.round(4).to_frame(name="Weight"))
st.markdown(f"**Consistency Ratio (CR):** {cr:.4f} " + ("✅ Acceptable" if cr < 0.1 else "⚠️ Review inputs"))

# -------------------- TOPSIS MATRIX --------------------
st.header("4️⃣ Enter Performance Matrix (TOPSIS)")

performance_data = []
for alt in alternatives:
    row = []
    for crit in criteria:
        val = st.number_input(f"{alt} – {crit}", value=1.0, step=0.1,
                              key=f"perf_{alt}_{crit}")
        row.append(val)
    performance_data.append(row)

decision_matrix = pd.DataFrame(performance_data, columns=criteria, index=alternatives)

st.subheader("Define Criteria Type")
criterion_types = {}
for crit in criteria:
    ctype = st.radio(f"{crit} is a...", ["Benefit (more is better)", "Cost (less is better)"],
                     key=f"type_{crit}")
    criterion_types[crit] = (ctype == "Benefit (more is better)")

# -------------------- TOPSIS CALCULATION --------------------
st.header("5️⃣ TOPSIS Ranking")

norm = decision_matrix.copy()
for col in norm.columns:
    denom = np.sqrt((norm[col]**2).sum())
    if denom == 0:
        norm[col] = 0
    else:
        norm[col] = norm[col] / denom

weighted = norm * weights.values

ideal = weighted.copy()
anti_ideal = weighted.copy()
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

results = pd.DataFrame({
    "Alternative": alternatives,
    "TOPSIS Score": scores_clean,
    "Rank": ranks
}).sort_values("TOPSIS Score", ascending=False).reset_index(drop=True)

st.subheader("Final Ranking")
st.dataframe(results)

fig = px.bar(results, x="Alternative", y="TOPSIS Score", color="TOPSIS Score",
             color_continuous_scale="Aggrnyl", text_auto=True)
fig.update_layout(yaxis_title="Score", xaxis_title="Alternative")
st.plotly_chart(fig, use_container_width=True)

st.success("🎉 Done! You can now compare results from multiple users.")