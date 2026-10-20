import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Credit Card Fraud Intelligence", page_icon="💳", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.metric-card {border:1px solid rgba(128,128,128,.25); border-radius:12px; padding:16px; background:rgba(128,128,128,.06);}
.small-note {color:#777; font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Training models and preparing dashboard data...")
def build_pipeline(csv_path):
    df = pd.read_csv(csv_path)
    x = df.drop(columns=["Class"])
    y = df["Class"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    x_train = x_train.copy()
    x_test = x_test.copy()

    # Same feature engineering as the supplied project.
    x_train["Hour"] = (x_train["Time"] % 86400) // 3600
    x_test["Hour"] = (x_test["Time"] % 86400) // 3600
    x_train["Day"] = x_train["Time"] // 86400
    x_test["Day"] = x_test["Time"] // 86400
    # Raw (unscaled) hour index and day, kept only for display; the models use the scaled columns.
    raw_train = x_train[["Hour", "Day"]].copy()
    raw_test = x_test[["Hour", "Day"]].copy()
    x_train.drop(columns=["Time"], inplace=True)
    x_test.drop(columns=["Time"], inplace=True)

    q1 = x_train["Amount"].quantile(0.25)
    q3 = x_train["Amount"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    x_train["Amount"] = x_train["Amount"].clip(lower, upper)
    x_test["Amount"] = x_test["Amount"].clip(lower, upper)

    scaler_amount = StandardScaler()
    scaler_time = StandardScaler()
    x_train[["Amount"]] = scaler_amount.fit_transform(x_train[["Amount"]])
    x_test[["Amount"]] = scaler_amount.transform(x_test[["Amount"]])
    x_train[["Hour", "Day"]] = scaler_time.fit_transform(x_train[["Hour", "Day"]])
    x_test[["Hour", "Day"]] = scaler_time.transform(x_test[["Hour", "Day"]])

    models = {
        "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
        "Logistic Regression": LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    }

    predictions = {}
    probabilities = {}
    metrics = []
    cms = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        prob = model.predict_proba(x_test)[:, 1]
        predictions[name] = pred
        probabilities[name] = prob
        metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
        })
        cms[name] = confusion_matrix(y_test, pred)

    test_display = x_test.copy()
    # Keep raw, human-readable values alongside the model-ready features.
    test_display["Original Amount"] = df.loc[x_test.index, "Amount"]
    test_display["Original Time"] = df.loc[x_test.index, "Time"]
    test_display["Actual Class"] = y_test
    test_display["Raw Hour"] = raw_test["Hour"]
    test_display["Raw Day"] = raw_test["Day"]
    for name in models:
        test_display[f"{name} Prediction"] = predictions[name]
        test_display[f"{name} Fraud Probability"] = probabilities[name]

    train_display = x_train.copy()
    train_display["Class"] = y_train
    train_display["Raw Hour"] = raw_train["Hour"]
    train_display["Raw Day"] = raw_train["Day"]

    return {
        "df": df,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "models": models,
        "metrics": pd.DataFrame(metrics),
        "cms": cms,
        "test_display": test_display,
        "train_display": train_display,
        "predictions": predictions,
        "probabilities": probabilities,
        "scaler_amount": scaler_amount,
        "scaler_time": scaler_time,
    }

st.title("💳 Credit Card Fraud Detection Intelligence")
st.caption("Interactive analysis of transaction patterns, fraud risk, and ML model performance")

with st.sidebar:
    st.header("Dashboard Controls")
    csv_path = st.text_input("Dataset path", "creditcard.csv")
    st.divider()
    st.write("**Project pipeline**")
    st.write("Train/test split → time features → IQR capping → scaling → three classifiers")

try:
    d = build_pipeline(csv_path)
except FileNotFoundError:
    st.error(f"Could not find `{csv_path}`. Put creditcard.csv in the same folder as dashboard.py, or change the Dataset path in the sidebar.")
    st.stop()
except Exception as e:
    st.exception(e)
    st.stop()

# ---------- KPI row ----------
df = d["df"]
metrics = d["metrics"]
fraud_count = int(df["Class"].sum())
normal_count = int((df["Class"] == 0).sum())
fraud_rate = fraud_count / len(df)
best_model = metrics.sort_values("F1 Score", ascending=False).iloc[0]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Transactions", f"{len(df):,}")
c2.metric("Fraudulent Transactions", f"{fraud_count:,}")
c3.metric("Fraud Rate", f"{fraud_rate * 100:.3f}%")
c4.metric("Best F1", f"{best_model['F1 Score']:.3f}")
c5.metric("Best Model", best_model["Model"])

st.info("Fraud is a severely imbalanced class in this dataset. Accuracy alone is therefore not a sufficient model-selection criterion; pay particular attention to recall, precision, and F1.")

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔎 Fraud Patterns", "🤖 Model Performance", "🚨 Transaction Inspector"])

with tab1:
    st.subheader("Dataset overview")
    left, right = st.columns(2)

    with left:
        class_df = pd.DataFrame({"Class": ["Normal", "Fraud"], "Transactions": [normal_count, fraud_count]})
        fig = px.bar(class_df, x="Class", y="Transactions", text="Transactions", title="Transaction Class Distribution")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        amount_fig = px.histogram(df, x="Amount", color=df["Class"].map({0:"Normal",1:"Fraud"}), nbins=60,
                                  title="Transaction Amount Distribution", labels={"color":"Class"}, barmode="overlay")
        amount_fig.update_traces(opacity=0.65)
        st.plotly_chart(amount_fig, use_container_width=True)

    st.subheader("Fraud probability by hour")
    train = d["train_display"].copy()
    hourly = train.groupby("Raw Hour", as_index=False)["Class"].mean().rename(columns={"Class":"Fraud Probability", "Raw Hour":"Hour"})
    hourly["Hour"] = hourly["Hour"].astype(int)
    fig = px.line(hourly, x="Hour", y="Fraud Probability", markers=True, title="Observed Fraud Probability by Hour")
    fig.update_yaxes(tickformat=".2%")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset quality")
    q1, q2, q3 = st.columns(3)
    q1.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")
    q2.metric("Features", f"{df.shape[1]-1}")
    q3.metric("Rows in Test Set", f"{len(d['y_test']):,}")

with tab2:
    train = d["train_display"].copy()

    st.subheader("When does fraud happen?")
    col1, col2 = st.columns(2)
    with col1:
        fraud_hour = train.groupby("Raw Hour", as_index=False)["Class"].sum().rename(columns={"Class":"Fraud Count", "Raw Hour":"Hour"})
        fraud_hour["Hour"] = fraud_hour["Hour"].astype(int)
        fig = px.bar(fraud_hour, x="Hour", y="Fraud Count", title="Fraud Transactions by Hour")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fraud_day = train.groupby("Raw Day", as_index=False)["Class"].sum().rename(columns={"Class":"Fraud Count", "Raw Day":"Day"})
        fraud_day["Day"] = fraud_day["Day"].astype(int)
        fig = px.bar(fraud_day, x="Day", y="Fraud Count", title="Fraud Transactions by Day")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Behavioral differences")
    feature = st.selectbox("Select feature", ["Amount", "V4", "V11", "V14"])
    behavior = train[[feature, "Class"]].copy()
    behavior["Class"] = behavior["Class"].map({0:"Normal", 1:"Fraud"})
    fig = px.box(behavior, x="Class", y=feature, color="Class", points=False,
                 title=f"{feature}: Normal vs Fraudulent Transactions")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature correlation")
    corr_cols = [c for c in d["x_train"].columns if c not in ["Hour", "Day"]]
    corr = d["x_train"][corr_cols].corr()
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu", zmid=0))
    fig.update_layout(height=650, title="Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Model comparison")
    metric = st.selectbox("Metric", ["F1 Score", "Recall", "Precision", "Accuracy"], index=0)
    fig = px.bar(metrics.sort_values(metric, ascending=False), x="Model", y=metric, text_auto=".3f",
                 title=f"{metric} Comparison")
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(metrics.style.format({"Accuracy":"{:.3f}", "Precision":"{:.3f}", "Recall":"{:.3f}", "F1 Score":"{:.3f}"}), use_container_width=True, hide_index=True)

    st.subheader("Confusion matrix")
    selected_model = st.selectbox("Select model", list(d["models"].keys()))
    cm = d["cms"][selected_model]
    cm_fig = px.imshow(cm, text_auto=True, x=["Normal", "Fraud"], y=["Normal", "Fraud"],
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       title=f"{selected_model} — Confusion Matrix")
    st.plotly_chart(cm_fig, use_container_width=True)

    st.subheader("What the metrics mean")
    st.markdown("""
    - **Precision:** Of transactions flagged as fraud, how many were actually fraud?
    - **Recall:** Of all actual fraud transactions, how many did the model catch?
    - **F1 Score:** Harmonic mean of precision and recall; useful when balancing false positives and false negatives.
    - **Accuracy:** Overall proportion of correct predictions; potentially misleading under severe class imbalance.
    """)

with tab4:
    st.subheader("Inspect a real held-out transaction")
    st.caption("This uses an actual transaction from the test split. It is not a synthetic manual form, because V1–V28 are PCA-transformed features rather than user-facing transaction attributes.")

    test_display = d["test_display"]
    row_numbers = list(range(len(test_display)))
    selected_idx = st.number_input("Test transaction index", min_value=0, max_value=max(0, len(row_numbers)-1), value=0, step=1)
    row = test_display.iloc[int(selected_idx)]

    a, b, c = st.columns(3)
    a.metric("Original Amount", f"{row['Original Amount']:.2f}")
    b.metric("Actual Class", "FRAUD" if row["Actual Class"] == 1 else "NORMAL")
    b.metric("Original Time", f"{row['Original Time']:.0f}")
    c.metric("Derived Hour", f"{int(row['Raw Hour'])}")

    st.markdown("### Model verdicts")
    verdict_cols = st.columns(3)
    for col, model_name in zip(verdict_cols, d["models"].keys()):
        pred = int(row[f"{model_name} Prediction"])
        prob = float(row[f"{model_name} Fraud Probability"])
        with col:
            if pred == 1:
                st.error(f"⚠ {model_name}: FRAUD")
            else:
                st.success(f"✓ {model_name}: NORMAL")
            st.metric("Fraud probability", f"{prob:.2%}")

    st.subheader("Transaction feature profile")
    feature_values = pd.DataFrame({"Feature": d["x_test"].columns, "Value": row[d["x_test"].columns].values})
    st.dataframe(feature_values, use_container_width=True, hide_index=True)

    st.download_button(
        "Download model comparison CSV",
        metrics.to_csv(index=False).encode("utf-8"),
        "model_comparison.csv",
        "text/csv"
    )

st.divider()
st.caption("Built from the supplied credit-card fraud detection pipeline. Model preprocessing follows the project code: stratified 80/20 split, Hour/Day feature engineering, IQR capping of Amount, and standard scaling.")
