# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Exercise Angle Prediction API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


# Global variables for models (will be loaded lazily)
model = None
label_encoder = None
side_encoder = None

def load_models():
    """Load models lazily when first needed"""
    global model, label_encoder, side_encoder
    
    if model is None:
        try:
            import joblib
            import numpy as np
            import pandas as pd
            
            logger.info("Loading models...")
            model = joblib.load("model/best_exercise_model.pkl")
            label_encoder = joblib.load("model/label_encoder.pkl")
            side_encoder = joblib.load("model/side_encoder.pkl")
            logger.info("Models and encoders loaded successfully")
            
        except ImportError as ie:
            logger.error(f"Import error: {ie}")
            raise HTTPException(status_code=500, detail=f"Import error: {str(ie)}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

# Define the expected input format
class ExerciseData(BaseModel):
    Side: str
    Shoulder_Angle: float
    Elbow_Angle: float
    Hip_Angle: float
    Knee_Angle: float
    Ankle_Angle: float
    Shoulder_Ground_Angle: float
    Elbow_Ground_Angle: float
    Hip_Ground_Angle: float
    Knee_Ground_Angle: float
    Ankle_Ground_Angle: float

    class Config:
        schema_extra = {
            "example": {
                "Side": "left",
                "Shoulder_Angle": 10.64,
                "Elbow_Angle": 174.47,
                "Hip_Angle": 174.79,
                "Knee_Angle": 175.00,
                "Ankle_Angle": 180.00,
                "Shoulder_Ground_Angle": 15.50,
                "Elbow_Ground_Angle": 25.30,
                "Hip_Ground_Angle": 45.20,
                "Knee_Ground_Angle": 90.15,
                "Ankle_Ground_Angle": 95.75
            }
        }

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Exercise Angle Prediction API",
        "version": "1.0.0",
        "supported_exercises": [
            "Jumping Jacks", "Squats", "Push Ups", "Pull ups", "Russian twists"
        ],
        "model_accuracy": "96.68%"
    }

@app.get("/health")
def health_check():
    try:
        load_models()
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        return {"status": "unhealthy", "model_loaded": False, "error": str(e)}

@app.post("/predict")
def predict(data: ExerciseData):
    try:
        # Load models if not already loaded
        load_models()
        
        # Import here to avoid circular import issues
        import pandas as pd
        import numpy as np
        
        # Create DataFrame with exact column order
        input_data = {
            'Side': [data.Side],
            'Shoulder_Angle': [data.Shoulder_Angle],
            'Elbow_Angle': [data.Elbow_Angle],
            'Hip_Angle': [data.Hip_Angle],
            'Knee_Angle': [data.Knee_Angle],
            'Ankle_Angle': [data.Ankle_Angle],
            'Shoulder_Ground_Angle': [data.Shoulder_Ground_Angle],
            'Elbow_Ground_Angle': [data.Elbow_Ground_Angle],
            'Hip_Ground_Angle': [data.Hip_Ground_Angle],
            'Knee_Ground_Angle': [data.Knee_Ground_Angle],
            'Ankle_Ground_Angle': [data.Ankle_Ground_Angle]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Encode the 'Side' column and rename it to match training
        try:
            input_df['Side_encoded'] = side_encoder.transform(input_df['Side'])
            # Drop the original 'Side' column
            input_df = input_df.drop('Side', axis=1)
        except ValueError as ve:
            logger.error(f"Side encoding error: {ve}")
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid 'Side' value. Expected: {list(side_encoder.classes_)}"
            )
        
        # Ensure column order (with Side_encoded instead of Side)
        expected_columns = [
            'Side_encoded', 'Shoulder_Angle', 'Elbow_Angle', 'Hip_Angle', 'Knee_Angle', 
            'Ankle_Angle', 'Shoulder_Ground_Angle', 'Elbow_Ground_Angle', 
            'Hip_Ground_Angle', 'Knee_Ground_Angle', 'Ankle_Ground_Angle'
        ]
        input_df = input_df[expected_columns]
        
        # Make prediction
        prediction_encoded = model.predict(input_df)[0]
        predicted_exercise = label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get confidence
        prediction_proba = model.predict_proba(input_df)[0]
        confidence = float(np.max(prediction_proba))
        
        return {
            "predicted_exercise": predicted_exercise,
            "confidence": round(confidence * 100, 2),
            "raw_prediction": int(prediction_encoded),
            "status": "success"
        }
        
    except ValueError as ve:
        logger.error(f"Value error in prediction: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid input data: {str(ve)}")
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-batch")
def predict_batch(data: list[ExerciseData]):
    """Predict multiple exercises at once"""
    try:
        results = []
        for item in data:
            single_result = predict(item)
            results.append(single_result)
        
        return {
            "predictions": results,
            "total_predictions": len(results),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model-info")
def get_model_info():
    """Get information about the loaded model"""
    try:
        load_models()
        
        feature_names = None
        if hasattr(model, 'feature_names_in_'):
            feature_names = list(model.feature_names_in_)
        
        return {
            "model_type": str(type(model)),
            "feature_names": feature_names,
            "side_encoder_classes": list(side_encoder.classes_),
            "label_encoder_classes": list(label_encoder.classes_),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
    
