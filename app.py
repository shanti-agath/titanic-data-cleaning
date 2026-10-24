"""
Customer Segmentation - Streamlit App
Run with:  streamlit run app.py
"""

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️")

INCOME = "Annual Income (k$)"
SPENDING = "Spending Score (1-100)"


def name_segment(income, spending):
    """Give a business-friendly name using income and spending levels."""
    if income >= 70 and spending >= 60:
        return "High income - High spending (Target)"
    if income >= 70 and spending < 40:
        return "High income - Low spending (Careful)"
    if income < 40 and spending >= 60:
        return "Low income - High spending (Impulsive)"
    if income < 40 and spending < 40:
        return "Low income - Low spending (Sensible)"
    return "Average income - Average spending (Standard)"


ADVICE = {
    "High income - High spending (Target)": "Best customers. Offer premium products and loyalty rewards.",
    "High income - Low spending (Careful)": "Can spend more. Use personalised offers to attract them.",
    "Low income - High spending (Impulsive)": "Spend a lot on a small income. Offer discounts and EMI deals.",
    "Low income - Low spending (Sensible)": "Price sensitive. Promote budget deals and combo offers.",
    "Average income - Average spending (Standard)": "Largest group. Keep them engaged with regular offers.",
}


@st.cache_resource
def train_model():
    """Train K-Means (k=5) on income and spending score."""
    df = pd.read_csv("Mall_Customers.csv")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[[INCOME, SPENDING]])

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    centres = df.groupby("Cluster")[[INCOME, SPENDING]].mean()
    names = {c: name_segment(r[INCOME], r[SPENDING]) for c, r in centres.iterrows()}
    df["Segment"] = df["Cluster"].map(names)
    return df, scaler, kmeans, names


df, scaler, kmeans, names = train_model()

st.title("🛍️ Customer Segmentation")
st.write("Find out which customer segment a person belongs to, "
         "using K-Means clustering on income and spending score.")

# ── Sidebar: user input ────────────────────────────
st.sidebar.header("Enter customer details")
income = st.sidebar.slider("Annual Income (k$)", 15, 140, 60)
spending = st.sidebar.slider("Spending Score (1-100)", 1, 100, 50)

# ── Predict segment ────────────────────────────────
new_point = pd.DataFrame([[income, spending]], columns=[INCOME, SPENDING])
cluster = int(kmeans.predict(scaler.transform(new_point))[0])
segment = names[cluster]

st.subheader("Predicted segment")
st.success(segment)
st.info(ADVICE[segment])

# ── Chart: all customers + the new customer ────────
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(x=INCOME, y=SPENDING, hue="Segment", data=df,
                palette="deep", s=50, ax=ax)
ax.scatter(income, spending, c="black", marker="*", s=400, label="New customer")
ax.set_title("Customer Segments")
ax.legend(fontsize=7, loc="upper right")
st.pyplot(fig)

# ── Segment summary table ──────────────────────────
st.subheader("Segment summary")
summary = df.groupby("Segment").agg(
    Customers=("CustomerID", "count"),
    Avg_Age=("Age", "mean"),
    Avg_Income=(INCOME, "mean"),
    Avg_Spending=(SPENDING, "mean"),
).round(1)
st.dataframe(summary)