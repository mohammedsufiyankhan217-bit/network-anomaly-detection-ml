import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
st.set_page_config(page_title="Network Anomaly Detection", page_icon="📊")


# Choose the dataset
st.subheader("Network Traffic Dataset")

data_option = st.radio(
    "Choose your data source:",
    ["Use existing dataset", "Upload a new CSV"],
    horizontal=True
)

if data_option == "Use existing dataset":
    df = pd.read_csv("dataset.csv")
else:
    uploaded_file = st.file_uploader(
        "Upload a network traffic CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("Please upload a CSV file to continue.")
        st.stop()


saved_data = joblib.load("network_model.pkl")
model = saved_data["model"]
model_features = saved_data["features"]
model_accuracy = round(saved_data["accuracy"], 2)
confusion = saved_data["confusion_matrix"]
report = saved_data["report"]



# Clean column names and labels
# Clean column names
df.columns = df.columns.str.strip()

# Clean labels only if the dataset contains them
if "Label" in df.columns:
    df["Label"] = df["Label"].astype(str).str.strip()

# Check whether the required ML features are available
missing_features = [
    feature for feature in model_features
    if feature not in df.columns
]

if missing_features:
    st.error(
        "This CSV is missing columns required by the ML model: "
        + ", ".join(missing_features[:10])
    )
    st.stop()

# Prepare data for machine learning predictions
prediction_features = df.reindex(columns=model_features).apply(
    pd.to_numeric, errors="coerce"
)

# Handle invalid values
prediction_features = prediction_features.replace(
    [float("inf"), float("-inf")], float("nan")
)

# Predict only rows with valid feature values
valid_rows = prediction_features.notna().all(axis=1)

df["ML Prediction"] = "Not predicted"

if valid_rows.any():
    predictions = model.predict(prediction_features.loc[valid_rows])

    df.loc[valid_rows, "ML Prediction"] = pd.Series(
        predictions,
        index=prediction_features.loc[valid_rows].index
    ).map({0: "BENIGN", 1: "DDoS"})


# -------------------------
# TITLE
# -------------------------

st.title("Network Anomaly Detection Dashboard")
st.write("CIC-IDS2017 Network Traffic Analysis")

# -------------------------
# SIDEBAR FILTER
# -------------------------


st.sidebar.header("Dashboard Filter")

if "Label" in df.columns:
    traffic_options = ["All Traffic"] + sorted(
        df["Label"].dropna().unique().tolist()
    )

    selected_traffic = st.sidebar.selectbox(
        "Select Traffic Type",
        traffic_options
    )
else:
    selected_traffic = "All Traffic"
    st.sidebar.info(
        "Uploaded data has no labels. Showing all traffic."
    )


# Apply filter
if selected_traffic == "All Traffic":
    filtered_df = df
else:
    filtered_df = df[df["Label"] == selected_traffic]

# -------------------------
# KEY NUMBERS
# -------------------------


total_records = len(filtered_df)

if "Label" in filtered_df.columns:
    normal_records = (filtered_df["Label"] == "BENIGN").sum()
    attack_records = (filtered_df["Label"] == "DDoS").sum()
else:
    normal_records = 0
    attack_records = 0


col1, col2, col3 = st.columns(3)

col1.metric("Total Traffic Records", f"{total_records:,}")
col2.metric("Normal Traffic", f"{normal_records:,}")
col3.metric("Anomalies / Attacks", f"{attack_records:,}")

# -------------------------
# TRAFFIC CLASSIFICATION
# -------------------------

st.subheader("Traffic Classification")


st.subheader("Traffic Type Distribution")

if "Label" in filtered_df.columns:
    label_counts = (
        filtered_df["Label"]
        .value_counts()
        .reset_index()
    )
    label_counts.columns = ["Traffic Type", "Count"]
    chart_title = "Traffic Type Distribution"
    category_column = "Traffic Type"
else:
    label_counts = (
        filtered_df["ML Prediction"]
        .value_counts()
        .reset_index()
    )
    label_counts.columns = ["Traffic Type", "Count"]
    chart_title = "ML-Predicted Traffic Distribution"
    category_column = "Traffic Type"

chart_type = st.selectbox(
    "Choose chart type for Traffic Classification",
    ["Bar", "Pie", "Line"]
)

if chart_type == "Bar":
    fig1 = px.bar(
        label_counts,
        x=category_column,
        y="Count",
        title=chart_title
    )
elif chart_type == "Pie":
    fig1 = px.pie(
        label_counts,
        names=category_column,
        values="Count",
        title=chart_title
    )
else:
    fig1 = px.line(
        label_counts,
        x=category_column,
        y="Count",
        title=chart_title,
        markers=True
    )

st.plotly_chart(fig1, use_container_width=True)

# -------------------------
# NORMAL VS ATTACK PIE CHART
# -------------------------


st.subheader("Normal vs Attack Traffic")

if "Label" in df.columns:
    overall_normal = (df["Label"] == "BENIGN").sum()
    overall_attack = (df["Label"] == "DDoS").sum()
    chart_title = "Overall Traffic Distribution"
else:
    overall_normal = (df["ML Prediction"] == "BENIGN").sum()
    overall_attack = (df["ML Prediction"] == "DDoS").sum()
    chart_title = "ML-Predicted Traffic Distribution"

pie_data = pd.DataFrame({
    "Category": ["Normal Traffic", "Attack Traffic"],
    "Count": [overall_normal, overall_attack]
})

fig2 = px.pie(
    pie_data,
    names="Category",
    values="Count",
    title=chart_title,
    hole=0.4
)

st.plotly_chart(fig2, use_container_width=True)


# -------------------------
# TOP DESTINATION PORTS
# -------------------------



st.subheader("Top Destination Ports")

if "Destination Port" in filtered_df.columns:
    port_counts = (
        filtered_df["Destination Port"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    port_counts.columns = ["Destination Port", "Count"]

    port_chart_type = st.selectbox(
        "Choose chart type for Destination Ports",
        ["Bar", "Pie", "Line"],
        key="port_chart_type"
    )

    if port_chart_type == "Bar":
        fig3 = px.bar(
            port_counts,
            x="Destination Port",
            y="Count",
            title="Top 10 Destination Ports"
        )
    elif port_chart_type == "Pie":
        fig3 = px.pie(
            port_counts,
            names="Destination Port",
            values="Count",
            title="Top 10 Destination Ports"
        )
    else:
        fig3 = px.line(
            port_counts,
            x="Destination Port",
            y="Count",
            title="Top 10 Destination Ports",
            markers=True
        )

    st.plotly_chart(fig3, use_container_width=True)

else:
    st.info("This CSV does not contain the Destination Port column.")


# -------------------------
# DATA TABLE
# -------------------------

st.subheader("Network Traffic Data")

st.caption("Showing the first 100 rows of the selected dataset.")
st.dataframe(filtered_df.head(100), use_container_width=True)




# -----------------------------
# ATTACK TRAFFIC BY TYPE
# -----------------------------


st.subheader("Attack Traffic by Type")

if "Label" in df.columns:
    attack_df = df[df["Label"] != "BENIGN"]

    attack_counts = (
        attack_df["Label"]
        .value_counts()
        .reset_index()
    )
    attack_counts.columns = ["Attack Type", "Count"]

    fig_attack = px.bar(
        attack_counts.head(10),
        x="Attack Type",
        y="Count",
        title="Top Attack Types"
    )
    st.plotly_chart(fig_attack, use_container_width=True)

else:
    predicted_ddos = (df["ML Prediction"] == "DDoS").sum()

    st.metric("Records predicted as DDoS", f"{predicted_ddos:,}")
    st.caption(
        "These are ML predictions, not confirmed attack types."
    )


# -----------------------------
# AVERAGE FLOW DURATION
# -----------------------------


st.subheader("Average Flow Duration by Traffic Type")

if "Flow Duration" in filtered_df.columns:
    if "Label" in filtered_df.columns:
        duration_df = (
            filtered_df.groupby("Label")["Flow Duration"]
            .mean()
            .reset_index()
        )
        duration_df.columns = ["Traffic Type", "Average Duration"]
    else:
        duration_df = pd.DataFrame({
            "Traffic Type": ["All Uploaded Traffic"],
            "Average Duration": [
                pd.to_numeric(
                    filtered_df["Flow Duration"],
                    errors="coerce"
                ).mean()
            ]
        })

    fig_duration = px.bar(
        duration_df,
        x="Traffic Type",
        y="Average Duration",
        title="Average Flow Duration"
    )

    st.plotly_chart(fig_duration, use_container_width=True)

else:
    st.info("This CSV does not contain the Flow Duration feature.")


st.divider()

st.header("Machine Learning Predictions")

st.metric("Model Test Accuracy", f"{model_accuracy}%")

st.write("Predictions generated by the trained Random Forest model.")

# Count the model predictions
st.subheader("ML Predictions on Selected Dataset")

predicted_normal = (df["ML Prediction"] == "BENIGN").sum()
predicted_ddos = (df["ML Prediction"] == "DDoS").sum()
not_predicted = (df["ML Prediction"] == "Not predicted").sum()

# Display prediction metrics
col1, col2, col3 = st.columns(3)

col1.metric("Predicted Normal Traffic", f"{predicted_normal:,}")
col2.metric("Predicted DDoS Attacks", f"{predicted_ddos:,}")
col3.metric("Rows Not Predicted", f"{not_predicted:,}")

# Visualize ML predictions
prediction_counts = (
    df["ML Prediction"]
    .value_counts()
    .rename_axis("Prediction")
    .reset_index(name="Number of Records")
)

fig_ml = px.pie(
    prediction_counts,
    names="Prediction",
    values="Number of Records",
    title="Traffic Predictions by ML Model",
    hole=0.4
)

st.plotly_chart(fig_ml, use_container_width=True)

st.subheader("Original Model Test Results — Confusion Matrix")

confusion_df = pd.DataFrame(
    confusion,
    index=["Actual BENIGN", "Actual DDoS"],
    columns=["Predicted BENIGN", "Predicted DDoS"]
)

st.dataframe(confusion_df)

st.caption(
    "Rows represent actual traffic classes; columns represent model predictions."
)

st.subheader("Original Model Test Results — Classification Metrics")

metric_col1, metric_col2, metric_col3 = st.columns(3)

metric_col1.metric(
    "DDoS Precision",
    f"{report['DDoS']['precision'] * 100:.2f}%"
)

metric_col2.metric(
    "DDoS Recall",
    f"{report['DDoS']['recall'] * 100:.2f}%"
)

metric_col3.metric(
    "DDoS F1-Score",
    f"{report['DDoS']['f1-score'] * 100:.2f}%"
)




