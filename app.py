import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "crop_disease_model.h5")
CLASS_PATH = os.path.join(BASE_DIR, "class_names.json")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- APP ----------------
app = Flask(__name__)
CORS(app)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully")

# ---------------- LOAD CLASSES ----------------
with open(CLASS_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files["image"]
    image = Image.open(file).convert("RGB")

    processed = preprocess_image(image)
    preds = model.predict(processed)

    class_index = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100

    return jsonify({
        "prediction": CLASS_NAMES[class_index],
        "confidence": round(confidence, 2)
    })

# ---------------- RUN ----------------
if __name__ == "__main__":
    import webbrowser
    webbrowser.open("http://127.0.0.1:5000/")
    app.run(debug=True)
