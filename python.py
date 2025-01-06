from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the saved model
model = joblib.load("pricing_model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Pricing Model API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    # Get JSON data from the request
    data = request.get_json()

    # Convert data into a DataFrame for the model
    df = pd.DataFrame([data])
    features = ["Price", "Competitor_Price", "Customer_Rating", "Demand_Elasticity"]

    try:
        # Make predictions
        prediction = model.predict(df[features])
        return jsonify({"Units_Sold": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    # Run the app on 0.0.0.0 to make it accessible from outside
    app.run(host="0.0.0.0", port=5000)
