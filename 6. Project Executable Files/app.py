from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
print(f"BASE_DIR: {BASE_DIR}")

TEMPLATE_PATH = os.path.join(BASE_DIR, "templates")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

print(f"MODEL_PATH: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

app = Flask(__name__, template_folder=TEMPLATE_PATH)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            request.form.get("Gender"), request.form.get("Place"),
            request.form.get("Type_of_alcohol"), request.form.get("Hepatitis_C"),
            request.form.get("Obesity"), request.form.get("Family_History"),
            request.form.get("TCH"), request.form.get("Hemoglobin"),
            request.form.get("PCV"), request.form.get("RBC"),
            request.form.get("MCV"), request.form.get("MCH"),
            request.form.get("MCHC"), request.form.get("Total_Count"),
            request.form.get("Polymorphs"), request.form.get("Lymphocytes"),
            request.form.get("Monocytes"), request.form.get("Eosinophils"),
            request.form.get("Basophils"), request.form.get("Platelet_Count"),
            request.form.get("Indirect"), request.form.get("Total_Protein"),
            request.form.get("Albumin"), request.form.get("Globulin"),
            request.form.get("A_G_Ratio"), request.form.get("USG_Abdomen"),
            request.form.get("AST_ALT_Ratio"), request.form.get("Direct_Total_Bilirubin_Ratio"),
            request.form.get("TG_HDL_Ratio"), request.form.get("LDL_HDL_Ratio"),
            request.form.get("RDW"), request.form.get("Lifetime_Alcohol_Consumption"),
            request.form.get("Alcohol_Consumption_Intensity"),
            request.form.get("Liver_Stress_Score"), request.form.get("Metabolic_Syndrome_Indicator"),
            request.form.get("FamilyHistory_Diabetes"), request.form.get("Obesity_LDL"),
            request.form.get("Obesity_TG"), request.form.get("Age_Category_Middle_Aged"),
            request.form.get("Age_Category_Young"), request.form.get("Alcohol_Category_Light"),
            request.form.get("Alcohol_Category_Moderate"), request.form.get("Alcohol_Category_Heavy"),
            request.form.get("BP_Category_Prehypertension"), request.form.get("BP_Category_Hypertension")
        ]

        features = np.array(features, dtype=float).reshape(1, -1)
        
        prediction = model.predict(features)[0]
        result = "Patient is at risk of Liver Cirrhosis" if prediction == 0 else "No Cirrhosis detected"

        return render_template("index.html", prediction_text=result)
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)