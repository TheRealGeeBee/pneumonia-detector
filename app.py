#----------------------------------------------------------------------------------
# THIS IS A PROJECT THAT USES AI TO CHECK FOR PNEUMONIA INFECTION USING XRAY IMAGES
#----------------------------------------------------------------------------------

import tensorflow as tf
from io import BytesIO
from flask import Flask, render_template, request
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

app = Flask(__name__)

uploaded_image = None


def predict_with_model(img):
    BASE_DIR = Path(__file__).resolve().parent.parent
    model_path = BASE_DIR / "Pneumonia Identification Using X-Ray Images (PhD Thesis)"/ "model" / "pneumonia_model_sigmoid.h5"

    resized_img = resize(img, (150, 150))
    final_shaped_img = resized_img[np.newaxis, ...]

    my_model = tf.keras.models.load_model(model_path)

    prediction = my_model.predict(final_shaped_img)[0][0]

    if prediction >= 0.5:
        label = "PNEUMONIA"
        confidence = float(prediction)
    else:
        label = "NORMAL"
        confidence = float(1 - prediction)

    return label, confidence


def redirect_to_results(prediction, confidence):

    if prediction == "PNEUMONIA":
        description = (
            "The model detected patterns consistent with pneumonia. "
            "Please consult a qualified medical professional as soon as possible "
            "for a full diagnosis and appropriate treatment."
        )
    else:
        description = (
            "The model did not detect signs of pneumonia in the uploaded X-ray. "
            "If you are experiencing symptoms such as persistent cough, fever, "
            "or difficulty breathing, you should still seek medical advice."
        )

    return render_template(
        "results.html",
        prediction=prediction,
        confidence=round(confidence * 100, 2),
        description=description
    )


@app.route("/", methods=["POST", "GET"])
def homepage():
    if request.method == "POST":
        global uploaded_image

        if 'image' not in request.files:
            return 'No file part'

        file = request.files['image']

        if file.filename == '':
            return 'No selected file'

        if file:
            img_bytes = file.read()
            uploaded_image = plt.imread(BytesIO(img_bytes), format=".jpg")

            pneumonia_class, confidence = predict_with_model(uploaded_image)

            return redirect_to_results(pneumonia_class, confidence)

    else:
        return render_template("home_page.html")


if __name__ == "__main__":
    app.run(debug=True)