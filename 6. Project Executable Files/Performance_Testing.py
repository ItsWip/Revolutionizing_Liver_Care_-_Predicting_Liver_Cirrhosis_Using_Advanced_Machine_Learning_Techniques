import pickle
import time
import numpy as np
import pandas as pd
from memory_profiler import memory_usage
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

df_test = pd.read_csv("Dataset/processed_dataset.csv") 
X_test = df_test.drop(columns=["Predicted Value(Out Come-Patient suffering from liver cirrosis or not)"])
y_test = df_test["Predicted Value(Out Come-Patient suffering from liver cirrosis or not)"]

start_time = time.time()
predictions = model.predict(X_test)
end_time = time.time()
print(f"Model Inference Time: {end_time - start_time:.4f} seconds")

def model_inference():
    return model.predict(X_test)

mem_usage = memory_usage(model_inference)
print(f"Memory Usage: {max(mem_usage):.2f} MB")

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)

print("\nModel Performance Metrics:")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1-Score: {f1:.4f}")
print(f"✅ ROC-AUC Score: {roc_auc:.4f}")

def make_prediction(input_data):
    return model.predict([input_data])

inputs = [X_test.iloc[i] for i in range(len(X_test))]

start_time = time.time()
with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(make_prediction, inputs))
end_time = time.time()

print(f"\nStress Test: Processed {len(results)} predictions in {end_time - start_time:.4f} seconds.")

conf_matrix = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix:")
print(conf_matrix)
