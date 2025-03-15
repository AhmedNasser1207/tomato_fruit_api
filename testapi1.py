import numpy as np
from tensorflow import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import matplotlib.pyplot as plt
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    try:
        # Validate file
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type")
        if file.size > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large")
        
        # Read and preprocess image
        contents = await file.read()
        class_num , conf = predict_disease(contents)
        return {"class_number": str(class_num), "confidence": float(conf)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to Object Detection API!"}

# Load the saved model
model = load_model("tomato_fruit_3.keras")

# Define class labels
class_labels = [
    "Class1", "Class2", "Class3", "Class4", "Class5"
    # Add your actual class names here
]

def predict_disease(image_content):
    # Load and preprocess the image
    img = image.decode_image(image_content, channels=3)
    img = image.resize(img, (244, 244))  # Match model input size
    img  = img.numpy()
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    
    # Make prediction
    predictions = model.predict(img)
    
    # Get top prediction
    predicted_class = np.argmax(predictions, axis=1)
    confidence = np.max(predictions) * 100
    
    return class_labels[predicted_class[0]], confidence

def test_single_image(image_path):
    # Get predictions
    predicted_label, confidence = predict_disease(image_path)
    
    # Display image with predictions
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Prediction: {predicted_label}\nConfidence: {confidence:.2f}%")
    plt.show()
    
    # Print detailed predictions
    print(f"Predicted Disease: {predicted_label}")
    print(f"Confidence: {confidence:.2f}%")
