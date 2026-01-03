# 🎗️ Breast Cancer Prediction Dashboard
📖 Overview
<img width="1894" height="939" alt="Screenshot 2026-01-03 123422" src="https://github.com/user-attachments/assets/511ea907-9afb-4ec7-b623-ab0fd0cd1aac" />
<img width="1883" height="934" alt="Screenshot 2026-01-03 123447" src="https://github.com/user-attachments/assets/610042b7-2d44-40db-9318-f13be0fd3c5e" />


The Breast Cancer Prediction Dashboard is an interactive Machine Learning web application built using Streamlit.
It uses a Logistic Regression model trained on the Breast Cancer Wisconsin dataset to predict whether a tumor is Benign or Malignant based on user-provided medical measurements.

The application combines ML prediction, data visualization, and model evaluation into a single, easy-to-use dashboard suitable for learning, demonstration, and portfolio purposes.

🚀 Key Features

🧠 Machine Learning Prediction – Logistic Regression classifier

🔍 Interactive Patient Input – Sliders for medical feature values

🩺 Real-time Prediction Output – Benign or Malignant classification

📊 Visual Analytics Dashboard – Histograms, scatter plots, heatmaps

📈 Model Evaluation – Accuracy, confusion matrix, ROC curve & AUC

📄 Dataset Exploration – Summary statistics and raw data view

🖥️ User-Friendly UI – Clean, tab-based Streamlit interface

🛠️ Tech Stack

Frontend & App Framework: Streamlit

Programming Language: Python

Machine Learning: Scikit-learn (Logistic Regression)

Data Handling: Pandas, NumPy

Visualization: Matplotlib, Seaborn

📂 Project Structure
project/
│── app.py               # Main Streamlit application
│── requirements.txt     # Project dependencies
│── README.md            # Documentation

▶️ How to Run

Install dependencies

pip install -r requirements.txt


Run the Streamlit app

streamlit run app.py


Use the sidebar sliders to enter patient data and click Predict

📊 Model Details

Algorithm: Logistic Regression

Data Preprocessing: StandardScaler

Evaluation Metrics:

Accuracy

Confusion Matrix

ROC Curve & AUC

💡 Use Cases

Medical ML demonstration project

Learning classification models

Data visualization with Streamlit

Resume & portfolio project

🔮 Future Enhancements

Add more ML models (Random Forest, SVM)

Deploy on Streamlit Cloud

Feature importance visualization

Patient report download option

📜 License

This project is licensed under the MIT License – free to use and modify.
