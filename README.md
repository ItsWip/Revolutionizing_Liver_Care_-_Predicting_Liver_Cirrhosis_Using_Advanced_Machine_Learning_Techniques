# Revolutionizing Liver Care: Predicting Liver Cirrhosis Using Advanced Machine Learning Techniques

## Table of Contents
- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Project Overview](#project-overview)
- [Technology Stack](#technology-stack)
- [Data Pipeline](#data-pipeline)
- [Model Training & Evaluation](#model-training--evaluation)
- [Results](#results)
- [Deployment](#deployment)
- [Future Scope](#future-scope)
- [How to Use](#how-to-use)
- [Contributors](#contributors)

## Introduction
Liver cirrhosis is a severe and progressive condition caused by long-term liver damage. If left undiagnosed, it can lead to life-threatening complications. This project leverages machine learning to predict liver cirrhosis risk and assist healthcare professionals with early intervention strategies.

## Problem Statement
Traditional diagnostic methods rely on symptomatic evaluation and expensive imaging techniques, often delaying early intervention. Our goal is to build a predictive model that identifies high-risk patients early, improving healthcare efficiency and patient outcomes.

## Project Overview
This project involves the development of a machine learning model to predict liver cirrhosis using medical history, lab test results, and lifestyle factors. The model classifies patients into different risk categories to aid in proactive treatment planning.

## Technology Stack
- **Programming Language**: Python
- **Data Handling**: Pandas, NumPy, SciPy
- **Machine Learning Frameworks**: Scikit-learn, XGBoost, TensorFlow/PyTorch
- **Model Deployment**: Flask (as API layer), Pickle (for saving model)
- **Visualization**: Matplotlib, Seaborn
- **Development Tools**: Jupyter Notebook, VS Code

## Data Pipeline
1. **Data Acquisition**: Medical records including lab tests, imaging results, and patient history.
2. **Data Preprocessing**: Cleaning, handling missing values, feature selection.
3. **Feature Engineering**: Identifying key biomarkers relevant to liver cirrhosis.
4. **Model Training**: Training on labeled datasets using various ML algorithms.
5. **Model Evaluation**: Using accuracy, precision, recall, F1-score, and ROC-AUC metrics.
6. **Deployment**: Deploying the model using Flask for real-time predictions.

## Model Training & Evaluation
The machine learning models tested include:
- **Random Forest** (Final Model)
- Logistic Regression
- Decision Tree
- XGBoost
- Support Vector Machine

### Performance Metrics:
- **Accuracy**: 99.78%
- **Precision**: 99.57%
- **Recall**: 100%
- **F1-score**: 99.78%
- **ROC-AUC Score**: 99.85%

## Results
The Random Forest model provided high accuracy and robustness, making it the best choice for this project. The model performed well in detecting high-risk patients with minimal false negatives.

## Deployment
The model is deployed as a Flask-based web API, allowing integration with healthcare systems.

### Steps to Run the Project:
1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/Liver-Cirrhosis-Prediction.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask app:
   ```bash
   python app.py
   ```

## Future Scope
- **Optimization**: Implementing feature selection techniques to reduce unnecessary features.
- **Alternative Models**: Testing Gradient Boosting models like LightGBM and CatBoost.
- **Scalability**: Deploying the model using Docker for seamless integration.
- **Real-World Applications**: Integration with Electronic Health Records (EHR) for automated risk assessment.

## Contributors
- **Swaraj Patil** (Project Lead)
- **Prisha Kumar**
- **Prajakta Padwalkar**
- **Atharva Chikane**

---
### Additional Resources:
- Dataset: [Kaggle Link](https://www.kaggle.com/datasets/bhavanipriya222/liver-cirrhosis-prediction)
- Project Repository: [GitHub Link](https://github.com/ItsWip/Revolutionizing_Liver_Care_-_Predicting_Liver_Cirrhosis_Using_Advanced_Machine_Learning_Techniques)

