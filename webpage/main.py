import pickle
import numpy as np
import pandas as pd

from flask import Flask, render_template, request


# [=========={ Load Trained Model }==========]

model_path = 'model.pkl'

with open(model_path, 'rb') as file:
    model = pickle.load(file)

# print("Model Loaded:", type(model))


app = Flask(__name__)


# [=========={ Home Page }==========]

@app.route("/")
def home():

    return render_template("index.html")


# [=========={ Prediction Route }==========]

@app.route("/predict", methods=['GET', 'POST'])
def predict():

    # CORRECTED: Allow direct GET access to /predict
    if request.method == 'GET':
        return render_template("index.html")


    # [=========={ Fetch Form Data }==========]

    Age = request.form.get('age', '').strip()

    Gender = request.form.get('Gender', '').strip()

    Daily_Usage_Hours = request.form.get('Daily_Usage_Hours', '').strip()

    Sleep_Duration_Hours = request.form.get('Sleep_Duration_Hours', '').strip()


    # [=========={ Input Validation }==========]

    # CORRECTED: Check empty fields

    if not all([
        Age,
        Gender,
        Daily_Usage_Hours,
        Sleep_Duration_Hours
    ]):

        return render_template(
            "index.html",
            error="Please fill all fields!"
        )


    # CORRECTED: Convert numerical inputs safely

    try:

        Age = float(Age)

        Daily_Usage_Hours = float(Daily_Usage_Hours)

        Sleep_Duration_Hours = float(Sleep_Duration_Hours)

    except ValueError:

        return render_template(
            "index.html",
            error="Please enter valid numerical values!"
        )


    # CORRECTED: Reject invalid numerical values

    if not all(np.isfinite([
        Age,
        Daily_Usage_Hours,
        Sleep_Duration_Hours
    ])):

        return render_template(
            "index.html",
            error="Invalid numerical values!"
        )


    # CORRECTED: Basic range validation

    if Age <= 0 or Daily_Usage_Hours < 0 or Sleep_Duration_Hours < 0:

        return render_template(
            "index.html",
            error="Please enter valid positive values!"
        )

    if Daily_Usage_Hours > 24 or Sleep_Duration_Hours > 24:

        return render_template(
            "index.html",
            error="Hours cannot exceed 24!"
        )


    # [=========={ Create DataFrame }==========]

    features = {

        'Age': [Age],

        'Gender': [Gender],

        'Daily_Usage_Hours': [Daily_Usage_Hours],

        'Sleep_Duration_Hours': [Sleep_Duration_Hours]

    }

    final_features = pd.DataFrame(features)


    # [=========={ Preprocessing }==========]

    # CORRECTED:
    # Manual encoding instead of calling
    # transform() on LinearRegression.

    # Assumes training encoding:
    # Female = 0
    # Male = 1

    gender_mapping = {

        'Female': 0,

        'Male': 1

    }

    final_features['Gender'] = final_features['Gender'].map(
        gender_mapping
    )


    # Check unknown categories

    if final_features['Gender'].isnull().any():

        return render_template(
            "index.html",
            error="Invalid gender selected!"
        )


    # [=========={ Prediction }==========]

    prediction = model.predict(final_features)


    # CORRECTED:
    # Assumes target encoding:
    # Beneficial = 0
    # Normal = 1

    output = ('Beneficial' if prediction[0] < 0.5 else 'Normal')


    # [=========={ Display Results }==========]

    print("\nReceived Features:")
    print(final_features)

    print("\nPrediction:")
    print(prediction)

    print("\nOutput:")
    print(output)


    return render_template(
        "index.html",
        prediction=output,
        raw_prediction=prediction[0]
    )


# [=========={ About Page }==========]

@app.route("/about")
def about():

    return render_template("about.html")


# [=========={ Run Flask }==========]

if __name__ == '__main__':

    app.run(debug=True)