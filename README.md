# Heart Disease Risk Prediction ❤️

A full-stack Machine Learning application designed to predict the likelihood of heart failure or stroke based on medical attributes. This project demonstrates the complete Data Science lifecycle, from Exploratory Data Analysis (EDA) and Model Comparison to a deployed **Streamlit Web App** for real-time risk assessment.

## 🚀 Project Overview

Cardiovascular diseases (CVDs) are a leading cause of death globally. This tool leverages patient data—such as age, cholesterol levels, and exercise-induced angina—to identify potential heart risks early. The goal is to provide a simple, accessible interface for preliminary heart health screening.

## 📂 Project Structure

The repository is organized as follows:

```text
Heart-Disease-Prediction/
│
├── app.py                       # 🖥️ Frontend: Streamlit Web Application
├── Heart_Disease_Project.ipynb  # 📓 Backend: EDA, Model Training & Comparison
├── heart.csv                    # 📊 Dataset: Raw medical records used for training
├── KNN_heart.pkl                # 📦 Artifact: Trained K-Nearest Neighbors Classifier
├── scaler.pkl                   # ⚖️ Artifact: Saved StandardScaler for normalization
├── columns.pkl                  # 📝 Artifact: List of columns for consistent One-Hot Encoding
├── requirements.txt             # ⚙️ Dependencies: Required Python libraries
└── README.md                    # 📄 Documentation


📊 Workflow & Features
1. Exploratory Data Analysis (Jupyter Notebook)
The project begins with a deep dive into the dataset using pandas, matplotlib, and seaborn:
Data Cleaning: Handled missing values and outliers in features like Cholesterol and RestingBP.
Visualization: Analyzed correlations between age, max heart rate, and heart disease presence.
Feature Engineering: Applied One-Hot Encoding to categorical variables (e.g., ChestPainType, RestingECG) to make them machine-readable.

2. Model Selection & Training
I trained and evaluated 5 different Machine Learning models to determine the best performer:
Logistic Regression (Accuracy: ~86.9%)
Support Vector Machine (SVM) (Accuracy: ~84.8%)
Decision Tree (Accuracy: ~76.1%)
Naive Bayes (Accuracy: ~84.8%)
K-Nearest Neighbors (KNN) (Accuracy: ~86.4%) ✅

Conclusion: The KNN model was selected for the final application due to its high accuracy and balanced F1-score (0.88), making it reliable for this classification task.

3. Web Application (Streamlit)
The frontend is built with Streamlit to allow for easy user interaction:
User-Friendly Interface: Intuitive sliders and dropdowns for entering medical details (e.g., Age, Sex, Cholesterol).
Robust Preprocessing: The app loads the saved scaler.pkl and columns.pkl to process user input exactly as the model expects, handling missing categorical features dynamically.
Real-Time Prediction: Instantly classifies the patient as "High Risk" or "Low Risk" using the pre-trained KNN model.

🛠️ Tech Stack
Language: Python 3.13.3
Machine Learning: Scikit-Learn
Web Framework: Streamlit
Data Processing: Pandas, NumPy
Visualization: Seaborn, Matplotlib
