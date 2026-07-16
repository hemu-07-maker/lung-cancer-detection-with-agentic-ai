from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model if it exists, otherwise use scoring logic
MODEL_PATH = "model/lung_cancer_model.pkl"

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

def score_risk(data):
    """
    Fallback scoring when no trained model is present.
    Weights based on Random Forest feature importances.
    """
    score = 0
    age = int(data.get("age", 45))

    if age >= 60: score += 20
    elif age >= 45: score += 10

    weights = {
        "smoking": 25,
        "chest_pain": 12,
        "shortness_of_breath": 10,
        "yellow_fingers": 8,
        "wheezing": 8,
        "swallowing_difficulty": 9,
        "coughing": 7,
        "chronic_disease": 6,
        "fatigue": 5,
        "alcohol": 4,
        "anxiety": 3,
        "peer_pressure": 3,
        "allergy": 2,
    }

    for key, weight in weights.items():
        if int(data.get(key, 1)) == 2:
            score += weight

    if int(data.get("gender", 0)) == 1:
        score += 3

    return min(score, 100)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if model:
            features = np.array([[
                int(data.get("gender", 0)),
                int(data.get("age", 45)),
                int(data.get("smoking", 1)),
                int(data.get("yellow_fingers", 1)),
                int(data.get("anxiety", 1)),
                int(data.get("peer_pressure", 1)),
                int(data.get("chronic_disease", 1)),
                int(data.get("fatigue", 1)),
                int(data.get("allergy", 1)),
                int(data.get("wheezing", 1)),
                int(data.get("alcohol", 1)),
                int(data.get("coughing", 1)),
                int(data.get("shortness_of_breath", 1)),
                int(data.get("swallowing_difficulty", 1)),
                int(data.get("chest_pain", 1)),
            ]])
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1] * 100
            risk_score = round(probability, 1)
        else:
            risk_score = score_risk(data)
            prediction = 1 if risk_score >= 50 else 0

        if risk_score < 30:
            level = "low"
        elif risk_score < 60:
            level = "moderate"
        else:
            level = "high"

        return jsonify({
            "risk_score": risk_score,
            "prediction": int(prediction),
            "risk_level": level,
            "message": "Assessment complete"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
