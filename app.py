import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageFilter

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "crop_disease_model.h5")
CLASS_PATH = os.path.join(BASE_DIR, "class_names.json")

# ================= APP =================
app = Flask(__name__)
CORS(app)

# ================= LOAD MODEL =================
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully")

# ================= LOAD CLASSES =================
with open(CLASS_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

# ================= IMAGE PREPROCESS =================
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ================= FARMER-REALISTIC LEAF CHECK =================
def is_valid_crop_leaf(image):
    img = np.array(image)

    if img.ndim != 3 or img.shape[2] != 3:
        return False

    # --- GREEN DOMINANCE CHECK ---
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    green_pixels = (g > r) & (g > b) & (g > 45)
    green_ratio = np.sum(green_pixels) / green_pixels.size

    # --- TEXTURE CHECK (leaf has texture, wall/logo doesn't) ---
    gray = image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_variance = np.var(np.array(edges))

    return green_ratio > 0.12 and edge_variance > 15

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({
            "prediction": "No Image",
            "confidence": 0,
            "message": "Please upload a crop leaf image"
        }), 400

    try:
        image = Image.open(request.files["image"]).convert("RGB")
    except:
        return jsonify({
            "prediction": "Invalid Image",
            "confidence": 0,
            "message": "Unsupported image format"
        }), 400

    # 🚫 Reject non-crop images
    if not is_valid_crop_leaf(image):
        return jsonify({
            "prediction": "Not a Crop Leaf",
            "confidence": 0,
            "message": "Upload a clear leaf of crops like maize, wheat, rice, tomato, potato etc."
        })

    processed = preprocess_image(image)
    preds = model.predict(processed)

    confidence = float(np.max(preds)) * 100
    class_index = int(np.argmax(preds))

    # 🚫 Low confidence
    if confidence < 55:
        return jsonify({
            "prediction": "Uncertain",
            "confidence": round(confidence, 2),
            "message": "Leaf image is unclear or disease symptoms are weak"
        })

    return jsonify({
        "prediction": CLASS_NAMES[class_index],
        "confidence": round(confidence, 2)
    })

# ================= RUN =================
if __name__ == "__main__":
    import webbrowser
    webbrowser.open("http://127.0.0.1:5000/")
    app.run(debug=True)
