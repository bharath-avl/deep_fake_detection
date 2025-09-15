# Deep_fake_detection
This repository contains trained deepfake detection models built using Machine Learning techniques. The models are trained on benchmark datasets and are capable of distinguishing between real and manipulated (deepfake) images/videos with high accuracy.

**Key Features**
Trained Model (.h5) – Pre-trained deepfake detection model ready for deployment.

Flask Integration – Easily integrate the model into a Flask-based web application for real-time deepfake detection.

Image/Video Input – Supports uploaded media files for analysis.

Output – Provides a prediction whether the given media is real or deepfake.

**Tech Stack**
Frameworks/Libraries: TensorFlow / Keras, OpenCV, NumPy, Flask

Model Format: .h5 (can be converted to other formats if needed)

Languages: Python (backend + ML integration), HTML/CSS/JavaScript (frontend, if using web interface)

**Usage**

Clone the repository.

Place the trained .h5 model inside the model/ directory.

Run the Flask app (python app.py).

Upload an image/video and get predictions.

**Project Structure**

deepfake-detector/
│── model/              # Trained .h5 deepfake detection models
│── app.py              # Flask application for serving the model
│── requirements.txt    # Dependencies
│── static/             # Frontend assets (CSS, JS)
│── templates/          # HTML templates for the web app

**Applications**

Detecting manipulated media on social media platforms

Enhancing cybersecurity and digital forensics

Preventing misinformation and fraud

Assisting journalism and law enforcement
