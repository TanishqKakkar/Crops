from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle
import os
from io import BytesIO

# Import weather/crop advisory functions from city.py
from city import (
    get_weather_by_city,
    check_disease_risks_all,
    check_stress_conditions_all,
    get_general_plant_health
)

backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(backend_dir)

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATHS = {
    "apple": {
        "cnn": "models/cnn/apple_disease_model.h5",
        "label_encoder": "PKL/label encoder/apple_cnn_label_encoder.pkl",
        "fusion": "models/fusion/apple_fusion_model.h5",
        "feature_scaler": "PKL/feature scale/apple_feature_scaler.pkl",
        "climate_cnn": "models/climate dependent cnn/apple_climate_disease_model.h5"
    },
    "cotton": {
        "cnn": "models/cnn/cotton_disease_model2.h5",
        "label_encoder": "PKL/label encoder/cotton_cnn_label_encoder.pkl",
        "fusion": "models/fusion/cotton_fusion_model.h5",
        "feature_scaler": "PKL/feature scale/cotton_feature_scaler.pkl",
        "climate_cnn": "models/climate dependent cnn/cotton_climate_disease_model.h5"
    },
    "maize": {
        "cnn": "models/cnn/maize_disease_model.h5",
        "label_encoder": "PKL/label encoder/maize_label_encoder.pkl",
        "fusion": "models/fusion/maize_fusion_model.h5",
        "feature_scaler": "PKL/feature scale/maize_feature_scaler.pkl",
        "climate_cnn": "models/climate dependent cnn/maize_climate_disease_model.h5"
    },
    "potato": {
        "cnn": "models/cnn/potato_disease_model.h5",
        "label_encoder": "PKL/label encoder/potato_cnn_label_encoder.pkl",
        "fusion": "models/fusion/potato_fusion_model.h5",
        "feature_scaler": "PKL/feature scale/potato_feature_scaler.pkl",
        "climate_cnn": "models/climate dependent cnn/potato_climate_disease_model.h5"
    },
    "tomato": {
        "cnn": "models/cnn/tomato_disease_model_finetuned.h5",
        "label_encoder": "PKL/label encoder/tomato_label_encoder.pkl",
        "fusion": "models/fusion/tomato_fusion_model.h5",
        "feature_scaler": "PKL/feature scale/tomato_feature_scaler.pkl",
        "climate_cnn": "models/climate dependent cnn/tomato_climate_disease_model.h5"
    }
}

CROP_CLASS_NAMES = {
    "apple": [
        "Alternaria leaf spot",
        "Brown spot",
        "Health",
        "Mosaic",
        "Powdery mildew",
        "Rust",
        "Scab"
    ],
    "maize": [
        "Common_rust",
        "Gray_leaf_spot",
        "Healthy",
        "Northern_Leaf_Blight"
    ],
    "tomato": [
        "Tomato_Bacterial_spot",
        "Tomato_Early_blight", 
        "Tomato_healthy",
        "Tomato_Late_blight",
        "Tomato_Leaf_Mold",
        "Tomato_Septoria_leaf_spot"
    ],
    "cotton": [
        "Aphids",
        "Army worm",
        "Bacterial blight",
        "Powdery mildew"
    ],
    "potato": [
        "Early_Blight",
        "Healthy",
        "Late_Blight"
    ]
}

# Apple CROP_DETAILS example, expand as needed for other crops
CROP_DETAILS = {
    "apple": [
        {
            "name": "Alternaria leaf spot",
            "symptoms": {
                "english": [
                    "Dark brown to black spots with concentric rings on leaves",
                    "Yellow halos around leaf spots",
                    "Premature leaf drop",
                    "Spots merging to form large dead areas",
                    "Fruit skin lesions or blemishes"
                ],
                "hindi": [
                    "पत्तियों पर गहरे भूरे से काले छल्लेदार धब्बे",
                    "धब्बों के चारों ओर पीले घेरे",
                    "समय से पहले पत्तियों का झड़ना",
                    "धब्बों का मिलकर बड़े मृत क्षेत्र बनाना",
                    "फलों की त्वचा पर धब्बे या दाग"
                ]
            },
            # ...similarly for causes, prevention, treatment
        },
        # Other apple diseases...
    ]
}

def load_model(path):
    absolute_path = os.path.join(project_root, path)
    return tf.keras.models.load_model(absolute_path)

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = image_array / 255.0
    return np.expand_dims(image_array, axis=0)

@app.post("/predict")
async def predict(
    crop: str = Form(...),
    image: UploadFile = File(...),
):
    crop = crop.strip().lower()
    if crop not in MODEL_PATHS:
        return {"error": "Invalid crop type"}
    if crop not in CROP_CLASS_NAMES:
        return {"error": "Class names not defined for this crop"}
    paths = MODEL_PATHS[crop]
    contents = await image.read()
    img = Image.open(BytesIO(contents))
    processed_image = preprocess_image(img)
    try:
        cnn_model = load_model(paths["cnn"])
        cnn_pred = cnn_model.predict(processed_image)
        cnn_pred_class_index = int(np.argmax(cnn_pred, axis=1)[0])
        class_names = CROP_CLASS_NAMES[crop]
        if cnn_pred_class_index < 0 or cnn_pred_class_index >= len(class_names):
            return {"error": f"Predicted class index {cnn_pred_class_index} is out of range for crop {crop}"}
        cnn_pred_class_name = class_names[cnn_pred_class_index]
        cnn_pred_class_name_disp = cnn_pred_class_name.replace("_", " ")
    except Exception as e:
        return {"error": f"Error during CNN prediction: {e}"}

    # Get details
    details = None
    if crop == "apple":
        for disease in CROP_DETAILS["apple"]:
            if disease["name"].lower() == cnn_pred_class_name_disp.lower():
                details = disease
                break

    response = {
        "crop": crop,
        "prediction": cnn_pred_class_name_disp,
        "model_used": "cnn",
        "details": details
    }
    return response

@app.get("/city_weather")
def city_weather(city: str = Query(..., description="City name")):
    try:
        weather_data, location, country, state = get_weather_by_city(city)
        temp = weather_data["temp"]
        humidity = weather_data["humidity"]
        health_status, health_message = get_general_plant_health(temp, humidity)
        crops = ['tomato', 'potato', 'maize', 'cotton', 'apple']

        crop_data = {}
        for crop in crops:
            stresses = check_stress_conditions_all(weather_data, crop)
            risks = check_disease_risks_all(weather_data, crop)
            formatted_risks = [
                {
                    "name": r["disease"],
                    "risk": r["risk"],
                    "probability": r["probability"],
                    "expected_range": r["expected_range"],
                    "advice": r["advice"]
                }
                for r in risks
            ]
            crop_data[crop] = {
                "plant_health": {
                    "status": health_status,
                    "message": health_message
                },
                "stresses": stresses,
                "diseases": formatted_risks
            }

        return {
            "temperature": temp,
            "humidity": humidity,
            "location": {
                "city": location,
                "country": country,
                "state": state
            },
            "crops": crop_data
        }
    except Exception as e:
        return {"error": f"Internal server error: {str(e)}"}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Crop Disease Prediction API"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
