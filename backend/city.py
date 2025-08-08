from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests

app = FastAPI()

# Enable CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = "c8e8ee68cb50840df4baec312f679a1f"  # Your OpenWeatherMap API key

# ===== DISEASES dictionary for all crops =====
DISEASES = {
    'tomato': [
        {"name": "Two-Spotted Spider Mites", "temp_range": (32, 36), "humidity_range": (30, 40),
         "advice": ["Increase humidity to deter mites.", "Spray neem oil or insecticidal soap weekly.",
                    "Introduce natural predators like predatory mites."]},
        {"name": "Bacterial spot", "temp_range": (25, 30), "humidity_range": (75, 90),
         "advice": ["Apply copper-based sprays at first sign of disease.", "Avoid overhead watering and use disease-free seeds.",
                    "Rotate crops with non-solanaceous plants."]},
        {"name": "Early_Blight", "temp_range": (24, 30), "humidity_range": (70, 90),
         "advice": ["Remove lower leaves and plant debris.", "Apply fungicides like mancozeb or chlorothalonil.",
                    "Water plants at the base to avoid splashing leaves."]},
        {"name": "Healthy", "temp_range": None, "humidity_range": None,
         "advice": ["Monitor regularly for signs of pests and diseases.", "Maintain proper fertilization and irrigation.",
                    "Practice crop rotation and field sanitation."]},
        {"name": "Late_Blight", "temp_range": (15, 22), "humidity_range": (90, 100),
         "advice": ["Use certified healthy seedlings.", "Apply fungicides like mancozeb at onset of cool, wet weather.",
                    "Remove and destroy infected leaves promptly."]},
        {"name": "Leaf Mold", "temp_range": (18, 25), "humidity_range": (85, 100),
         "advice": ["Improve air circulation and avoid wetting leaves.", "Apply copper-based fungicides when symptoms appear.",
                    "Space plants adequately to reduce humidity."]},
        {"name": "Powdery mildew", "temp_range": (20, 25), "humidity_range": (50, 65),
         "advice": ["Spray with sulfur-based fungicides at first sign of powdery spots.", "Ensure good air flow around plants.",
                    "Remove and destroy affected plant parts."]},
        {"name": "Septoria leaf spot", "temp_range": (21, 27), "humidity_range": (85, 100),
         "advice": ["Remove and destroy infected leaves.", "Use fungicides such as chlorothalonil or copper compounds.",
                    "Keep foliage as dry as possible and use drip irrigation."]},
        {"name": "Tomato Yellow Leaf Curl Virus", "temp_range": (30, 34), "humidity_range": (40, 55),
         "advice": ["Use insect-proof netting to keep whiteflies out.",
                    "Regularly monitor and control whitefly populations.",
                    "Apply recommended insecticides such as imidacloprid if necessary."]}
    ],
    'potato': [
        {"name": "Early Blight", "temp_range": (18, 24), "humidity_range": (80, 95),
         "advice": ["Use resistant varieties.", "Spray fungicides at first sign of lesions.", "Remove and destroy affected leaves."]},
        {"name": "Late Blight", "temp_range": (10, 20), "humidity_range": (90, 100),
         "advice": ["Apply copper-based fungicides.", "Avoid dense foliage and improve airflow.", "Remove infected plants promptly."]},
        {"name": "Blackleg", "temp_range": (15, 22), "humidity_range": (85, 98),
         "advice": ["Plant certified seed potatoes.", "Ensure well-drained soil.", "Avoid over-watering."]},
    ],
    'maize': [
        {"name": "Common Rust", "temp_range": (15, 23), "humidity_range": (90, 100),
         "advice": ["Use resistant hybrids.", "Rotate maize with non-cereal crops.", "Timely fungicide application."]},
        {"name": "Gray Leaf Spot", "temp_range": (24, 32), "humidity_range": (75, 100),
         "advice": ["Remove crop debris after harvest.", "Apply fungicides as needed.", "Practice field sanitation."]},
        {"name": "Northern Leaf Blight", "temp_range": (18, 26), "humidity_range": (75, 95),
         "advice": ["Use tolerant varieties.", "Apply fungicides.", "Remove infected debris from field."]},
    ],
    'cotton': [
        {"name": "Aphids", "temp_range": (20, 35), "humidity_range": (40, 75),
         "advice": ["Monitor and use yellow sticky traps.", "Use neem oil spray.", "Release ladybird beetles."]},
        {"name": "Bacterial Blight", "temp_range": (28, 34), "humidity_range": (80, 100),
         "advice": ["Use disease-free seeds.", "Apply copper-based bactericides.", "Practice crop rotation."]},
        {"name": "Powdery Mildew", "temp_range": (18, 27), "humidity_range": (55, 70),
         "advice": ["Apply sulfur fungicides.", "Improve air circulation.", "Avoid late irrigations."]},
    ],
    'apple': [
        {"name": "Apple Scab", "temp_range": (12, 20), "humidity_range": (85, 100),
         "advice": ["Apply fungicides at green tip stage.", "Remove and destroy fallen leaves.", "Ensure orchard sanitation."]},
        {"name": "Powdery Mildew", "temp_range": (17, 24), "humidity_range": (65, 85),
         "advice": ["Prune out infected shoots.", "Apply recommended sulfur sprays.", "Maintain proper spacing."]},
        {"name": "Fire Blight", "temp_range": (24, 32), "humidity_range": (60, 90),
         "advice": ["Avoid overhead irrigation.", "Remove and burn infected branches.", "Spray copper compounds."]},
    ],
}

# ===== STRESS_CONDITIONS dictionary for all crops =====
STRESS_CONDITIONS = {
    'tomato': [
        {"name": "Heat Stress", "condition": lambda t, h: t > 34,
         "advice": ["Provide shade during peak sunlight hours.", "Water early in the morning or late in the evening.",
                    "Mulch to reduce soil temperature and retain moisture."]},
        {"name": "Drought Stress", "condition": lambda t, h: h < 35,
         "advice": ["Increase watering frequency.", "Use drip irrigation for water efficiency.",
                    "Apply organic mulch to retain soil moisture."]},
        {"name": "Cold Stress", "condition": lambda t, h: t < 10,
         "advice": ["Use covers to retain heat at night.", "Avoid fertilizing during cold spells.",
                    "Water during the day to warm up the soil."]},
        {"name": "Pest Vulnerability", "condition": lambda t, h: t > 32 and h < 40,
         "advice": ["Monitor for pests like mites and whiteflies.",
                    "Apply neem oil or insecticidal soap.",
                    "Raise humidity slightly to reduce pest activity."]}
    ],
    'potato': [
        {"name": "Heat Stress", "condition": lambda t, h: t > 28,
         "advice": ["Mulch potato beds.", "Shade nets if possible.", "Irrigate during early evening."]},
        {"name": "Drought Stress", "condition": lambda t, h: h < 40,
         "advice": ["Increase irrigation.", "Mulch to retain moisture."]},
        {"name": "Waterlogging", "condition": lambda t, h: h > 95,
         "advice": ["Improve drainage.", "Raised beds planting."]},
    ],
    'maize': [
        {"name": "Heat Stress", "condition": lambda t, h: t > 36,
         "advice": ["Water maize during heatwaves.", "Avoid planting during peak summer."]},
        {"name": "Drought Stress", "condition": lambda t, h: h < 35,
         "advice": ["Schedule irrigation regularly.", "Apply thick mulching."]},
    ],
    'cotton': [
        {"name": "Heat Stress", "condition": lambda t, h: t > 38,
         "advice": ["Increase frequency of irrigation.", "Avoid high dose fertilizers during heat."]},
        {"name": "Drought Stress", "condition": lambda t, h: h < 30,
         "advice": ["Irrigate cotton field at critical growth stages.", "Mulching between rows."]},
        {"name": "Waterlogging", "condition": lambda t, h: h > 90,
         "advice": ["Provide drainage.", "Avoid excessive irrigation."]},
    ],
    'apple': [
        {"name": "Cold Stress", "condition": lambda t, h: t < 5,
         "advice": ["Cover trees during spring frost.", "Use wind machines or heaters in orchards."]},
        {"name": "Drought Stress", "condition": lambda t, h: h < 35,
         "advice": ["Irrigate orchard regularly.", "Use basin irrigation & mulch."]},
        {"name": "Heat Stress", "condition": lambda t, h: t > 32,
         "advice": ["Whitewash trunks.", "Install shade net during summer peak."]},
    ],
}

# ==== Utility functions ====
def get_weather_by_city(city: str):
    geocode_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    res = requests.get(geocode_url)
    res.raise_for_status()
    data = res.json()
    lat = data["coord"]["lat"]
    lon = data["coord"]["lon"]
    country = data["sys"]["country"]
    state = data.get("state", "N/A")
    location_name = data["name"]
    return get_weather(lat, lon), location_name, country, state

def get_weather(lat: float, lon: float):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    res = requests.get(url)
    res.raise_for_status()
    data = res.json()
    return {
        "temp": data["main"]["temp"],
        "humidity": data["main"]["humidity"]
    }

def get_general_plant_health(temp: float, humidity: float):
    ideal_temp = (24, 30)
    ideal_humidity = (60, 80)
    temp_score = 100 - min(abs(temp - (ideal_temp[0] + ideal_temp[1]) / 2), 100)
    hum_score = 100 - min(abs(humidity - (ideal_humidity[0] + ideal_humidity[1]) / 2), 100)
    health_score = int((temp_score + hum_score) / 2)
    if health_score >= 80:
        return "Excellent", "🌱 Conditions are ideal for crop growth"
    elif health_score >= 60:
        return "Good", "🌿 Conditions are generally good for crops"
    elif health_score >= 40:
        return "Fair", "⚠ Conditions are suboptimal - monitor plants closely"
    else:
        return "Poor", "❌ Conditions are stressful for crops - take preventive measures"

def check_disease_risks_all(weather: dict, crop: str, temp_tolerance=5, hum_tolerance=5):
    risks = []
    temp = weather["temp"]
    humidity = weather["humidity"]
    crop_key = crop.strip().lower()
    if crop_key not in DISEASES:
        return risks
    for disease in DISEASES[crop_key]:
        if disease["temp_range"] is None or disease["humidity_range"] is None:
            continue
        t_low, t_high = disease["temp_range"]
        h_low, h_high = disease["humidity_range"]
        temp_match = (t_low - temp_tolerance <= temp <= t_high + temp_tolerance)
        hum_match = (h_low - hum_tolerance <= humidity <= h_high + hum_tolerance)
        temp_score = 100 - (abs(((t_low + t_high) / 2) - temp) / ((t_high - t_low) / 2) * 100)
        hum_score = 100 - (abs(((h_low + h_high) / 2) - humidity) / ((h_high - h_low) / 2) * 100)
        temp_score = max(0, min(temp_score, 100))
        hum_score = max(0, min(hum_score, 100))
        avg_score = int((temp_score + hum_score) / 2)
        if temp_match and hum_match:
            if avg_score >= 70:
                risk_level = "HIGH"
            elif avg_score >= 40:
                risk_level = "MODERATE"
            else:
                risk_level = "LOW"
        elif temp_match or hum_match:
            risk_level = "POTENTIAL"
        else:
            continue
        risks.append({
            "disease": disease["name"],
            "risk": risk_level,
            "probability": avg_score,
            "expected_range": f"Temp: {t_low}-{t_high} °C, Humidity: {h_low}-{h_high}%",
            "advice": disease["advice"]
        })
    return risks

def check_stress_conditions_all(weather: dict, crop: str):
    stresses = []
    temp = weather["temp"]
    humidity = weather["humidity"]
    crop_key = crop.strip().lower()
    if crop_key not in STRESS_CONDITIONS:
        return stresses
    for condition in STRESS_CONDITIONS[crop_key]:
        if condition["condition"](temp, humidity):
            stresses.append({
                "name": condition["name"],
                "advice": condition["advice"]
            })
    return stresses

# ==== /city_weather endpoint ====
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

        return JSONResponse({
            "temperature": temp,
            "humidity": humidity,
            "location": {
                "city": location,
                "country": country,
                "state": state
            },
            "crops": crop_data
        })

    except requests.exceptions.HTTPError:
        return JSONResponse({"error": "City not found"}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": f"Internal server error: {str(e)}"}, status_code=500)

# To run this FastAPI app:
# uvicorn city:app --reload
