import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px

st.set_page_config(page_title="Opioid Crisis Prediction & Response Tool", layout="wide")

st.title("Opioid Crisis Prediction & Response Tool")
st.markdown("A public health dashboard to identify vulnerable communities and prioritize intervention.")

@st.cache_data
def load_data():
    opioid = pd.read_csv("opioid_harms_mock.csv")
    health = pd.read_csv("bc_health_indicators.csv")
    return opioid, health

opioid, health = load_data()

# Clean column names
opioid.columns = opioid.columns.str.strip().str.lower()
health.columns = health.columns.str.strip().str.lower()

# Rename long column to a cleaner internal name
opioid = opioid.rename(columns={
    "apparent_opioid_toxicity_deaths": "apparent_opioid_tox"
})

# Sidebar
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Community Risk", "Forecast", "Response"]
)

# -----------------------------
# COMMUNITY RISK SCORE
# -----------------------------
risk_df = health.copy()

risk_features = [
    "opioid_overdose_rate",
    "mental_health_hospitalization_rate",
    "pct_smokers",
    "pct_heavy_drinkers",
    "pct_below_poverty_line",
    "pct_without_family_doctor",
    "diabetes_prevalence"
]

for col in risk_features:
    risk_df[col] = pd.to_numeric(risk_df[col], errors="coerce")

risk_df = risk_df.dropna(subset=risk_features)

risk_df["raw_risk_score"] = (
    0.25 * risk_df["opioid_overdose_rate"] +
    0.20 * risk_df["mental_health_hospitalization_rate"] +
    0.15 * risk_df["pct_smokers"] +
    0.10 * risk_df["pct_heavy_drinkers"] +
    0.15 * risk_df["pct_below_poverty_line"] +
    0.10 * risk_df["pct_without_family_doctor"] +
    0.05 * risk_df["diabetes_prevalence"]
)

# Normalize to 0-100
min_score = risk_df["raw_risk_score"].min()
max_score = risk_df["raw_risk_score"].max()

if max_score == min_score:
    risk_df["risk_score"] = 50
else:
    risk_df["risk_score"] = 100 * (
        (risk_df["raw_risk_score"] - min_score) / (max_score - min_score)
    )

def classify_risk(score):
    if score < 33:
        return "Low"
    elif score < 66:
        return "Medium"
    return "High"

risk_df["risk_level"] = risk_df["risk_score"].apply(classify_risk)

# -----------------------------
# OPIOID TREND PREP
# -----------------------------
opioid["year"] = pd.to_numeric(opioid["year"], errors="coerce")

quarter_map = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
opioid["quarter_num"] = opioid["quarter"].map(quarter_map)

opioid["apparent_opioid_tox"] = pd.to_numeric(opioid["apparent_opioid_tox"], errors="coerce")
opioid["opioid_hospitalizations"] = pd.to_numeric(opioid["opioid_hospitalizations"], errors="coerce")
opioid["opioid_ed_visits"] = pd.to_numeric(opioid["opioid_ed_visits"], errors="coerce")

trend_df = (
    opioid.groupby(["year", "quarter_num"], as_index=False)[
        ["apparent_opioid_tox", "opioid_hospitalizations", "opioid_ed_visits"]
    ]
    .sum()
    .sort_values(["year", "quarter_num"])
    .reset_index(drop=True)
)

trend_df["year"] = trend_df["year"].astype(int)
trend_df["quarter_num"] = trend_df["quarter_num"].astype(int)
trend_df["time_index"] = np.arange(len(trend_df))
trend_df["period_label"] = trend_df["year"].astype(str) + " Q" + trend_df["quarter_num"].astype(str)

# -----------------------------
# PAGE: OVERVIEW
# -----------------------------
if page == "Overview":
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Opioid Deaths", f"{int(trend_df['apparent_opioid_tox'].sum()):,}")
    col2.metric("Total Hospitalizations", f"{int(trend_df['opioid_hospitalizations'].sum()):,}")
    col3.metric("Total ED Visits", f"{int(trend_df['opioid_ed_visits'].sum()):,}")

    metric_labels = {
    "apparent_opioid_tox": "Apparent Opioid Toxicity Deaths",
    "opioid_hospitalizations": "Number of Opioid Hospitalizations",
    "opioid_ed_visits": "Number of Opioid ED visits",
    }
    
    metric_choice = st.selectbox(
    "Choose trend metric",
    options=list(metric_labels.keys()),
    format_func=lambda x: x
    )
    st.caption(f"**Label key:** `{metric_choice}` = {metric_labels[metric_choice]}")
        
    label = metric_labels.get(metric_choice, metric_choice)
    # line chart 
    fig = px.line(
        trend_df,
        x="period_label",
        y=metric_choice,
        markers=True,
        title=f"Trend Over Time: {label}",
        labels={ 
        "period_label": "Time",
        metric_choice: label
        }
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# PAGE: COMMUNITY RISK
# -----------------------------
elif page == "Community Risk":
    if "health_authority" in risk_df.columns:
        authorities = ["All"] + sorted(risk_df["health_authority"].dropna().unique().tolist())
        selected_authority = st.selectbox("Filter by Health Authority", authorities)

        display_df = risk_df.copy()
        if selected_authority != "All":
            display_df = display_df[display_df["health_authority"] == selected_authority]
    else:
        display_df = risk_df.copy()

    top_n = st.slider("Top communities to display", 5, 20, 10)

    top_risk = display_df.sort_values("risk_score", ascending=False).head(top_n)

    fig = px.bar(
        top_risk,
        x="chsa_name",
        y="risk_score",
        color="risk_level",
        title="Highest Risk Communities"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        display_df[
            [
                "chsa_name",
                "health_authority",
                "risk_score",
                "risk_level",
                "opioid_overdose_rate",
                "pct_below_poverty_line",
                "pct_without_family_doctor"
            ]
        ].sort_values("risk_score", ascending=False),
        use_container_width=True
    )

# -----------------------------
# PAGE: FORECAST
# -----------------------------
elif page == "Forecast":
    target = st.selectbox(
        "Select forecast target",
        ["apparent_opioid_tox", "opioid_ed_visits", "opioid_hospitalizations"]
    )

    X = trend_df[["time_index"]]
    y = trend_df[target]

    model = LinearRegression()
    model.fit(X, y)

    trend_df["predicted"] = model.predict(X)

    future_steps = 4
    future_idx = np.arange(
        trend_df["time_index"].max() + 1,
        trend_df["time_index"].max() + 1 + future_steps
    )
    future_pred = model.predict(future_idx.reshape(-1, 1))

    future_df = pd.DataFrame({
        "time_index": future_idx,
        "forecast": future_pred
    })

    fig_hist = px.line(
        trend_df,
        x="period_label",
        y=target,
        markers=True,
        title=f"Historical {target.replace('_', ' ').title()}"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Forecast for Next 4 Periods")
    st.dataframe(future_df, use_container_width=True)

    if y.iloc[-1] != 0:
        change = ((future_pred[-1] - y.iloc[-1]) / y.iloc[-1]) * 100
        st.info(f"Projected change over next 4 periods: {change:.1f}%")
    else:
        st.info("Projected change over next 4 periods could not be calculated because the latest value is 0.")

# -----------------------------
# PAGE: RESPONSE
# -----------------------------
elif page == "Response":
    response_df = risk_df[["chsa_name", "health_authority", "risk_score", "risk_level"]].copy()

    def recommendation(level):
        if level == "High":
            return "Prioritize naloxone distribution, outreach services, and addiction/mental health support."
        elif level == "Medium":
            return "Increase prevention efforts, screening, and access to primary care/community support."
        return "Maintain monitoring, prevention education, and routine support services."

    response_df["recommended_action"] = response_df["risk_level"].apply(recommendation)

    st.dataframe(
        response_df.sort_values("risk_score", ascending=False),
        use_container_width=True
    )