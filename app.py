import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

st.set_page_config(page_title="Breast Cancer Prediction Dashboard", layout="wide")

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
feature_names = data.feature_names

# ---------------------------------------------------
# Train Model
# ---------------------------------------------------
X = data.data
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.title("🎗️ Breast Cancer Prediction Dashboard")
st.write("A complete interactive ML app with visual analytics & prediction")

# Sidebar Inputs
st.sidebar.header("🔍 Enter Patient Details")

user_input = []
for f in feature_names:
    val = st.sidebar.slider(
        f, float(df[f].min()), float(df[f].max()), float(df[f].mean())
    )
    user_input.append(val)

# Prediction
if st.sidebar.button("Predict"):
    inp = np.array(user_input).reshape(1, -1)
    inp_scaled = scaler.transform(inp)

    pred = model.predict(inp_scaled)[0]
    prob = model.predict_proba(inp_scaled)[0][1]

    result = "Benign (SAFE)" if pred == 1 else "Malignant (CANCER)"

    st.markdown("## 🩺 Prediction Result")
    st.success(f"**Prediction: {result}**")
    st.info(f"**Probability of Benign: {prob:.4f}**")

# Tabs Layout
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset Overview",
    "📈 Visualizations",
    "🤖 Model Performance",
    "📝 Raw Data"
])

# ------------------- Dataset Overview -------------------
with tab1:
    st.subheader("Dataset Summary")
    st.write(df.describe())

    st.write("### Class Distribution")
    st.bar_chart(df["target"].value_counts())

# ------------------- Visualizations -------------------
with tab2:
    st.subheader("Visual Analytics Dashboard")

    col1, col2 = st.columns(2)

    # Histogram
    with col1:
        st.write("### Histogram")
        feature = st.selectbox("Select Feature", feature_names)
        fig, ax = plt.subplots(figsize=(5,4))
        sns.histplot(df[feature], kde=True, ax=ax)
        st.pyplot(fig)

    # Scatter Plot
    with col2:
        st.write("### Scatter Plot")
        f1 = st.selectbox("X-axis Feature", feature_names, key="f1")
        f2 = st.selectbox("Y-axis Feature", feature_names, key="f2")
        fig2, ax2 = plt.subplots(figsize=(5,4))
        sns.scatterplot(x=df[f1], y=df[f2], hue=df["target"], palette="Set1", ax=ax2)
        st.pyplot(fig2)

    # Heatmap
    st.write("### Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

# ------------------- Model Performance -------------------
with tab3:
    st.subheader("Model Evaluation Metrics")

    # Accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.metric(label="Model Accuracy", value=f"{accuracy*100:.2f}%")

    # Confusion Matrix
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    st.pyplot(fig_cm)

    # ROC Curve
    st.write("### ROC Curve")
    y_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax_roc.plot([0,1], [0,1], linestyle="--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")

# ------------------- RAW DATA -------------------
with tab4:
    st.subheader("Complete Dataset")
    st.dataframe(df)