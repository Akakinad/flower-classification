
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from .flower_names import FLOWER_NAMES

app = FastAPI(
    title="Flower Classification API",
    description="API for classifying 102 flower species using MobileNetV2",
    version="1.0.0"
)

print("Loading model...")
MODEL_PATH = "models/flower_classifier_mobilenetv2.keras"
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

IMAGE_SIZE = 224

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/")
def read_root():
    return {
        "message": "Flower Classification API is running!",
        "model": "MobileNetV2",
        "accuracy": "53.93%",
        "classes": 102
    }

@app.post("/predict")
async def predict_flower(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        processed_image = preprocess_image(image_bytes)
        predictions = model.predict(processed_image, verbose=0)
        top_5_indices = predictions[0].argsort()[-5:][::-1]
        
        results = []
        for idx in top_5_indices:
            results.append({
                "class_id": int(idx),
                "flower_name": FLOWER_NAMES[idx],
                "confidence": float(predictions[0][idx] * 100)
            })
        
        return JSONResponse(content={
            "success": True,
            "predictions": results,
            "filename": file.filename
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
