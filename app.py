import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Completion Dashboard", layout="wide")

# ---------- LOAD ----------
@st.cache_data
def load():
    df = pd.read_csv("Completion Rate by Source.csv")
    df["date"] = pd.to_datetime(df["source_click_dt"])
    return df

df = load()

# ---------- SIDEBAR ----------
st.sidebar.header("Filters")

date_range = st.sidebar.date_input(
    "Date range",
    [df["date"].min(), df["date"].max()]
)

sources = st.sidebar.multiselect(
    "Source",
    df["source"].unique(),
    default=df["source"].value_counts().head(10).index
)

clients = st.sidebar.multiselect(
    "Client",
    df["clientType"].unique(),
    default=df["clientType"].unique()
)

window = st.sidebar.radio("Completion window", ["Same day", "Within 3 days"])
metric_mode = st.sidebar.radio("Metric", ["Rate", "Absolute"])
rolling = st.sidebar.slider("Rolling avg", 1, 28, 7)

# ---------- FILTER ----------
mask = (
    (df["date"] >= pd.to_datetime(date_range[0])) &
    (df["date"] <= pd.to_datetime(date_range[1])) &
    (df["source"].isin(sources)) &
    (df["clientType"].isin(clients))
)
df_f = df[mask].copy()

# ---------- COLUMNS ----------
if window == "Same day":
    cols = [
        "ep2_same_day_users",
        "ep10_same_day_users",
        "ep20_same_day_users",
        "ep30_same_day_users",
        "ep50_same_day_users"
    ]
else:
    cols = [
        "ep2_3day_users",
        "ep10_3day_users",
        "ep20_3day_users",
        "ep30_3day_users",
        "ep50_3day_users"
    ]

# ---------- AGG ----------
daily = df_f.groupby(["date", "source"])[["total_user_series"] + cols].sum().reset_index()

if metric_mode == "Rate":
    for c in cols:
        daily[c] = daily[c] / daily["total_user_series"]

# ---------- ROLLING ----------
daily = daily.sort_values("date")
roll = (
    daily.set_index("date")
    .groupby("source")[cols]
    .rolling(rolling, min_periods=1)
    .mean()
    .reset_index()
)

st.title("📊 Source Completion Dashboard")

# ---------- SOURCE CONTRIBUTION ----------
st.subheader("🚦 Source Entry Contribution")
st.caption("Share of total users starting from each source within the selected filters.")

source_contrib = (
    df_f.groupby("source")["total_user_series"]
    .sum()
    .sort_values(ascending=False)
)

total_users = source_contrib.sum()
source_contrib_pct = (source_contrib / total_users).reset_index()
source_contrib_pct.columns = ["source", "share"]

# Show top contributors as metrics
top_sources = source_contrib_pct.head(5)

metric_cols = st.columns(len(top_sources))
for i, row in top_sources.iterrows():
    metric_cols[list(top_sources.index).index(i)].metric(
        row["source"],
        f"{row['share']:.1%}"
    )

# Bar chart
fig_contrib = px.bar(
    source_contrib_pct,
    x="source",
    y="share",
    title="Source share of total starts"
)
st.plotly_chart(fig_contrib, use_container_width=True)

st.divider()

# ---------- KPI ----------
st.subheader("Key Metrics")
st.caption("Average completion performance across the selected period and filters.")

kpi_cols = st.columns(len(cols))
for i, c in enumerate(cols):
    val = daily[c].mean()
    kpi_cols[i].metric(
        c.replace("_users",""),
        f"{val:.2%}" if metric_mode=="Rate" else f"{val:,.0f}"
    )

st.divider()

# ---------- TIME SERIES ----------
st.subheader("Trend Over Time")
st.caption("Shows how the selected completion metric evolves daily (rolling average applied).")

metric_select = st.selectbox("Metric", cols)

fig = px.line(
    roll,
    x="date",
    y=metric_select,
    color="source",
    title="Trend"
)
st.plotly_chart(fig, use_container_width=True)

# ---------- FUNNEL ----------
st.subheader("🎯 Funnel View")
st.caption("Shows average progression of users through episode milestones.")

funnel_vals = daily[cols].mean().values
funnel_labels = [c.replace("_users","") for c in cols]

fig_funnel = go.Figure(go.Funnel(y=funnel_labels, x=funnel_vals))
st.plotly_chart(fig_funnel, use_container_width=True)

# ---------- DROP-OFF ----------
st.subheader("📉 Step Conversion / Drop-off")
st.caption("Conversion rate between consecutive milestones and the percentage of users lost.")

drop_df = pd.DataFrame({
    "step": funnel_labels[:-1],
    "next_step": funnel_labels[1:],
    "conversion_rate": funnel_vals[1:] / funnel_vals[:-1]
})
drop_df["drop_off"] = 1 - drop_df["conversion_rate"]

st.dataframe(drop_df.style.format({
    "conversion_rate": "{:.2%}",
    "drop_off": "{:.2%}"
}))

# ---------- COHORT ----------
st.subheader("📅 Cohort Table")
st.caption("Average completion rates per source — useful to compare long-term quality of traffic.")

cohort = (
    daily.groupby("source")[cols]
    .mean()
    .sort_values(by=cols[0], ascending=False)
)
st.dataframe(cohort, use_container_width=True)

# ---------- STACKED CONTRIBUTION ----------
st.subheader("📊 Source Contribution Over Time")
st.caption("Shows how much each milestone contributes relative to total completions per day.")

stack = daily.groupby("date")[cols].sum()
stack_share = stack.div(stack.sum(axis=1), axis=0).reset_index()

metric_stack = st.selectbox("Metric for contribution", cols, key="stack")

fig_stack = px.area(
    stack_share,
    x="date",
    y=metric_stack,
    title="Source share over time"
)
st.plotly_chart(fig_stack, use_container_width=True)

# ---------- INSIGHTS ----------
st.subheader("🤖 Automatic Insights")
st.caption("Highlights statistically notable movements and anomalies.")

insights = []

latest = daily[daily["date"] >= daily["date"].max() - pd.Timedelta(days=7)]
prev = daily[
    (daily["date"] < daily["date"].max() - pd.Timedelta(days=7)) &
    (daily["date"] >= daily["date"].max() - pd.Timedelta(days=14))
]

if len(prev) > 0:
    change = (
        latest.groupby("source")[metric_select].mean()
        - prev.groupby("source")[metric_select].mean()
    ).dropna()

    if len(change) > 0:
        insights.append(f"📈 Biggest improvement: **{change.idxmax()}**")
        insights.append(f"📉 Biggest drop: **{change.idxmin()}**")

series = daily.groupby("date")[metric_select].mean()
z = (series - series.mean()) / series.std()

anomalies = series[np.abs(z) > 2]
if len(anomalies) > 0:
    insights.append(f"⚠️ {len(anomalies)} anomaly days detected")

if insights:
    for i in insights:
        st.write(i)
else:
    st.write("No major anomalies detected")

# ---------- DOWNLOAD ----------
st.download_button(
    "Download filtered data",
    df_f.to_csv(index=False),
    "filtered_data.csv"
)
