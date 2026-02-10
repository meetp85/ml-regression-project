from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    size = float(request.form["size"])
    prediction = model.predict(np.array([[size]]))[0]
    return render_template("index.html", prediction_text=f"Predicted Price: {prediction:.2f} Lakhs")

if __name__ == "__main__":
    app.run(debug=True)
