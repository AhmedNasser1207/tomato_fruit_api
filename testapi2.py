import numpy as np
import cv2
import requests
from io import BytesIO

from tensorflow import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse

import matplotlib.pyplot as plt

app = FastAPI()

# Load the saved model
model = load_model("tomato_fruit_3.keras")

# Define class labels
class_labels = [
    "Class1", "Class2", "Class3", "Class4", "Class5"
    # Replace with your actual class names
]

@app.get("/")
async def root():
    return {"message": "Welcome to Object Detection API!"}

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type")
        if file.size > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large")
        
        contents = await file.read()
        class_num, conf = predict_disease(contents)
        return {"class_number": str(class_num), "confidence": float(conf)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.get("/detect/url")
async def detect_from_url(image_url: str = Query(..., description="URL of the image to analyze")):
    try:
        response = requests.get(image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Unable to fetch image from URL")
        
        image_bytes = response.content
        class_num, conf = predict_disease(image_bytes)
        return {"class_number": str(class_num), "confidence": float(conf)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection from URL failed: {str(e)}")

def predict_disease(image_content):
    img = image.decode_image(image_content, channels=3)
    img = image.resize(img, (244, 244))  # Resize to model input
    img = img.numpy()
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)
    confidence = np.max(predictions) * 100

    return class_labels[predicted_class[0]], confidence
