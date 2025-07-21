# main.py
# To run this:
# 1. Ensure all libraries are installed from the virtual environment.
# 2. Ensure the 'deepfake-detection-model.h5' file is in the same directory.
# 3. Run from your terminal (inside venv): python main.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2  # OpenCV
from PIL import Image
import os
import io

# --- Initialization ---
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# --- Load the Machine Learning Model ---
MODEL_PATH = 'deepfake-detection-model.h5' 
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
    else:
        print(f"❌ Error: Model file not found at '{MODEL_PATH}'.")
        print("Please download the model and place it in the same directory as main.py.")
        model = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# --- Preprocessing Function ---
def preprocess_image(file_stream):
    """
    Reads an image from a file stream, preprocesses it for the model.
    """
    try:
        image = Image.open(file_stream).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image)
        image_array = image_array / 255.0
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None


# --- API Endpoint for Analysis ---
@app.route('/analyze', methods=['POST'])
def analyze_media():
    """
    This endpoint receives a file, preprocesses it, runs the ML model,
    and returns the deepfake prediction.
    """
    if model is None:
        return jsonify({"error": "Model is not available or failed to load."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file:
        try:
            file_stream = io.BytesIO(file.read())
            processed_image = preprocess_image(file_stream)
            
            if processed_image is None:
                 return jsonify({"error": "Could not process the image file."}), 400

            prediction = model.predict(processed_image)[0][0]
            
            # --- LOGIC FIX ---
            # Some models are trained where a LOW score means FAKE.
            # We are flipping the logic here to account for that.
            # Previously: is_fake = prediction > 0.5
            # Now: A lower score is more likely to be a deepfake.
            is_fake = prediction < 0.5
            confidence = (1.0 - float(prediction)) if is_fake else float(prediction)

            result = {
                "prediction": "Deepfake Detected" if is_fake else "Likely Authentic",
                "confidence": confidence
            }
            
            print(f"Analysis result: {result}")
            return jsonify(result)

        except Exception as e:
            print(f"An error occurred during analysis: {e}")
            return jsonify({"error": f"An error occurred: {e}"}), 500

    return jsonify({"error": "An unknown error occurred"}), 500

# --- Main entry point for the application ---
if __name__ == '__main__':
    # Runs the Flask app. Use the --port flag if you need to specify one.
    # Example: python main.py --port 5050
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5050, help='Port to run the server on')
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port, debug=True)

#I've updated the `main.py` file with the corrected logic for interpreting the model's predictions. After you replace the code in your file, please stop your old server and restart it. The analysis should now be much more accura