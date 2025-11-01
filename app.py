from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model only
model = joblib.load("Model/concrete_strength_xgb_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    input_data = {}
    
    if request.method == "POST":
        try:
            # Get user inputs
            input_data = {
                "cement": float(request.form["cement"]),
                "slag": float(request.form["slag"]),
                "flyash": float(request.form["flyash"]),
                "water": float(request.form["water"]),
                "superplasticizer": float(request.form["superplasticizer"]),
                "coarseagg": float(request.form["coarseagg"]),
                "fineagg": float(request.form["fineagg"]),
                "age": float(request.form["age"]),
            }

            # Convert input data into numpy array for prediction
            features = np.array([[v for v in input_data.values()]])

            # Predict compressive strength
            predicted_strength = model.predict(features)[0]
            prediction = round(predicted_strength, 2)

        except Exception as e:
            prediction = f"Error: {e}"

    # Render webpage with prediction and input values retained
    return render_template("index.html", prediction=prediction, input_data=input_data)

if __name__ == "__main__":
    app.run(debug=True)
