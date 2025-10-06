"""
EASI Severity Prediction REST API
==================================

FastAPI-based REST API for predicting EASI scores from dermatological images.
Designed for integration with Flutter mobile applications.

Endpoints:
- POST /predict - Upload image and get EASI predictions
- GET /health - Health check endpoint
- GET /conditions - Get list of available conditions

Installation:
pip install fastapi uvicorn python-multipart pillow tensorflow numpy pandas gdown

Run:
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import warnings
import logging
from typing import List, Dict, Any, Optional
from io import BytesIO
import base64
import zipfile
import shutil

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['MLIR_CRASH_REPRODUCER_DIRECTORY'] = ''
warnings.filterwarnings('ignore')
logging.getLogger('absl').setLevel(logging.ERROR)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image
import pickle
import pandas as pd
import gdown

# Initialize FastAPI app
app = FastAPI(
    title="EASI Severity Prediction API",
    description="REST API for predicting EASI scores from skin images",
    version="1.0.0"
)

# CORS middleware for Flutter web/mobile
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Flutter app domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration - UPDATE THESE WITH YOUR GOOGLE DRIVE FILE IDs
GDRIVE_DERM_FOUNDATION_ID = "1EW6cgnhE0yuFWmKNXr7gsuyoAOXWhO10"  # Replace with your file ID
DERM_FOUNDATION_PATH = "./derm_foundation/"

# Response Models
class ConditionPrediction(BaseModel):
    condition: str
    probability: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0)
    weight: float = Field(..., ge=0)
    easi_category: Optional[str] = None
    easi_contribution: int = Field(..., ge=0, le=3)

class EASIComponent(BaseModel):
    name: str
    score: int = Field(..., ge=0, le=3)
    contributing_conditions: List[Dict[str, Any]]

class PredictionResponse(BaseModel):
    success: bool
    total_easi_score: int = Field(..., ge=0, le=12)
    severity_interpretation: str
    easi_components: Dict[str, EASIComponent]
    predicted_conditions: List[ConditionPrediction]
    summary_statistics: Dict[str, float]
    image_info: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    available_conditions: int

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None


# Model wrapper class
class DermFoundationNeuralNetwork:
    def __init__(self):
        self.model = None
        self.mlb = None
        self.embedding_scaler = None
        self.confidence_scaler = None
        self.weighted_scaler = None
    
    def load_model(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.mlb = model_data['mlb']
            self.embedding_scaler = model_data['embedding_scaler']
            self.confidence_scaler = model_data['confidence_scaler']
            self.weighted_scaler = model_data['weighted_scaler']
            
            keras_model_path = model_data['keras_model_path']
            if os.path.exists(keras_model_path):
                self.model = tf.keras.models.load_model(keras_model_path)
                return True
            else:
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, embedding):
        if self.model is None:
            return None
        
        if len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)
        
        embedding_scaled = self.embedding_scaler.transform(embedding)
        predictions = self.model.predict(embedding_scaled, verbose=0)
        
        condition_probs = predictions['conditions'][0]
        individual_confidences = predictions['individual_confidences'][0]
        individual_weights = predictions['individual_weights'][0]
        
        condition_threshold = 0.3
        predicted_condition_indices = np.where(condition_probs > condition_threshold)[0]
        
        predicted_conditions = []
        predicted_confidences = []
        predicted_weights_dict = {}
        
        for idx in predicted_condition_indices:
            condition_name = self.mlb.classes_[idx]
            condition_prob = float(condition_probs[idx])
            
            if individual_confidences[idx] > 0:
                confidence_orig = self.confidence_scaler.inverse_transform([[individual_confidences[idx]]])[0, 0]
            else:
                confidence_orig = 0.0
                
            if individual_weights[idx] > 0:
                weight_orig = self.weighted_scaler.inverse_transform([[individual_weights[idx]]])[0, 0]
            else:
                weight_orig = 0.0
            
            predicted_conditions.append(condition_name)
            predicted_confidences.append(max(0, confidence_orig))
            predicted_weights_dict[condition_name] = max(0, weight_orig)
        
        all_condition_probs = {}
        all_confidences = {}
        all_weights = {}
        
        for i, class_name in enumerate(self.mlb.classes_):
            all_condition_probs[class_name] = float(condition_probs[i])
            
            if individual_confidences[i] > 0:
                conf_orig = self.confidence_scaler.inverse_transform([[individual_confidences[i]]])[0, 0]
                all_confidences[class_name] = max(0, conf_orig)
            else:
                all_confidences[class_name] = 0.0
                
            if individual_weights[i] > 0:
                weight_orig = self.weighted_scaler.inverse_transform([[individual_weights[i]]])[0, 0]
                all_weights[class_name] = max(0, weight_orig)
            else:
                all_weights[class_name] = 0.0
        
        return {
            'dermatologist_skin_condition_on_label_name': predicted_conditions,
            'dermatologist_skin_condition_confidence': predicted_confidences,
            'weighted_skin_condition_label': predicted_weights_dict,
            'all_condition_probabilities': all_condition_probs,
            'all_individual_confidences': all_confidences,
            'all_individual_weights': all_weights,
            'condition_threshold': condition_threshold
        }


# Helper function to download from Google Drive
def download_from_gdrive(file_id, output_path, is_folder=False):
    """Download file or folder from Google Drive"""
    try:
        print(f"Downloading from Google Drive (ID: {file_id})...")
        
        if is_folder:
            # For folders, download as zip
            url = f"https://drive.google.com/uc?id={file_id}"
            zip_path = output_path + ".zip"
            gdown.download(url, zip_path, quiet=False, fuzzy=True)
            
            # Extract zip
            print(f"Extracting to {output_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(output_path))
            
            os.remove(zip_path)
            print(f"Downloaded and extracted successfully")
        else:
            # For single files
            url = f"https://drive.google.com/uc?id={file_id}"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            gdown.download(url, output_path, quiet=False, fuzzy=True)
            print(f"Downloaded successfully to {output_path}")
        
        return True
    except Exception as e:
        print(f"Error downloading from Google Drive: {e}")
        return False


# EASI calculation functions
def calculate_easi_scores(predictions):
    easi_categories = {
        'erythema': {
            'name': 'Erythema (Redness)',
            'conditions': [
                'Post-Inflammatory hyperpigmentation', 'Erythema ab igne', 'Erythema annulare centrifugum',
                'Erythema elevatum diutinum', 'Erythema gyratum repens', 'Erythema multiforme',
                'Erythema nodosum', 'Flagellate erythema', 'Annular erythema', 'Drug Rash',
                'Allergic Contact Dermatitis', 'Irritant Contact Dermatitis', 'Contact dermatitis',
                'Acute dermatitis', 'Chronic dermatitis', 'Acute and chronic dermatitis',
                'Sunburn', 'Photodermatitis', 'Phytophotodermatitis', 'Rosacea',
                'Seborrheic Dermatitis', 'Stasis Dermatitis', 'Perioral Dermatitis',
                'Burn erythema of abdominal wall', 'Burn erythema of back of hand',
                'Burn erythema of lower leg', 'Cellulitis', 'Infection of skin',
                'Viral Exanthem', 'Infected eczema', 'Crusted eczematous dermatitis',
                'Inflammatory dermatosis', 'Vasculitis of the skin', 'Leukocytoclastic Vasculitis',
                'Cutaneous lupus', 'CD - Contact dermatitis', 'Acute dermatitis, NOS',
                'Herpes Simplex', 'Hypersensitivity', 'Impetigo', 'Pigmented purpuric eruption',
                'Pityriasis rosea', 'Tinea', 'Tinea Versicolor'
            ]
        },
        'induration': {
            'name': 'Induration/Papulation (Swelling/Bumps)',
            'conditions': [
                'Prurigo nodularis', 'Urticaria', 'Granuloma annulare', 'Morphea',
                'Scleroderma', 'Lichen Simplex Chronicus', 'Lichen planus', 'lichenoid eruption',
                'Lichen nitidus', 'Lichen spinulosus', 'Lichen striatus', 'Keratosis pilaris',
                'Molluscum Contagiosum', 'Verruca vulgaris', 'Folliculitis', 'Acne',
                'Hidradenitis', 'Nodular vasculitis', 'Sweet syndrome', 'Necrobiosis lipoidica',
                'Basal Cell Carcinoma', 'SCC', 'SCCIS', 'SK', 'ISK',
                'Cutaneous T Cell Lymphoma', 'Skin cancer', 'Adnexal neoplasm',
                'Insect Bite', 'Milia', 'Miliaria', 'Xanthoma', 'Psoriasis',
                'Lichen planus/lichenoid eruption'
            ]
        },
        'excoriation': {
            'name': 'Excoriation (Scratching Damage)',
            'conditions': [
                'Inflicted skin lesions', 'Scabies', 'Abrasion', 'Abrasion of wrist',
                'Superficial wound of body region', 'Scrape', 'Animal bite - wound',
                'Pruritic dermatitis', 'Prurigo', 'Atopic dermatitis', 'Scab'
            ]
        },
        'lichenification': {
            'name': 'Lichenification (Skin Thickening)',
            'conditions': [
                'Lichenified eczematous dermatitis', 'Acanthosis nigricans',
                'Hyperkeratosis of skin', 'HK - Hyperkeratosis', 'Keratoderma',
                'Ichthyosis', 'Ichthyosiform dermatosis', 'Chronic eczema',
                'Psoriasis', 'Xerosis'
            ]
        }
    }
    
    def probability_to_score(prob):
        if prob < 0.171:
            return 0
        elif prob < 0.238:
            return 1
        elif prob < 0.421:
            return 2
        elif prob < 0.614:
            return 3
        else:
            return 3
    
    easi_results = {}
    all_condition_probs = predictions['all_condition_probabilities']
    
    for component, category_info in easi_categories.items():
        category_conditions = []
        
        for condition_name, probability in all_condition_probs.items():
            if condition_name.lower() == 'eczema':
                continue
                
            if condition_name in category_info['conditions']:
                category_conditions.append({
                    'condition': condition_name,
                    'probability': probability,
                    'individual_score': probability_to_score(probability)
                })
        
        category_conditions = [c for c in category_conditions if c['individual_score'] > 0]
        category_conditions.sort(key=lambda x: x['probability'], reverse=True)
        
        component_score = sum(c['individual_score'] for c in category_conditions)
        component_score = min(component_score, 3)
        
        easi_results[component] = {
            'name': category_info['name'],
            'score': component_score,
            'contributing_conditions': category_conditions
        }
    
    total_easi = sum(result['score'] for result in easi_results.values())
    
    return easi_results, total_easi


def get_severity_interpretation(total_easi):
    if total_easi == 0:
        return "No significant EASI features detected"
    elif total_easi <= 3:
        return "Mild EASI severity"
    elif total_easi <= 6:
        return "Moderate EASI severity"
    elif total_easi <= 9:
        return "Severe EASI severity"
    else:
        return "Very Severe EASI severity"


# Image processing functions
def smart_crop_to_square(image):
    width, height = image.size
    if width == height:
        return image
    
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size
    
    return image.crop((left, top, right, bottom))


def generate_derm_foundation_embedding(model, image):
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        buf = BytesIO()
        image.save(buf, format='JPEG')
        image_bytes = buf.getvalue()
        
        input_tensor = tf.train.Example(features=tf.train.Features(
            feature={'image/encoded': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[image_bytes]))
            })).SerializeToString()
        
        infer = model.signatures["serving_default"]
        output = infer(inputs=tf.constant([input_tensor]))
        
        if 'embedding' in output:
            embedding_vector = output['embedding'].numpy().flatten()
        else:
            key = list(output.keys())[0]
            embedding_vector = output[key].numpy().flatten()
        
        return embedding_vector
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")


# Global model instances
derm_model = None
easi_model = None


@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global derm_model, easi_model
    
    # Download Derm Foundation model from Google Drive if not present
    if not os.path.exists(DERM_FOUNDATION_PATH) or not os.path.exists(os.path.join(DERM_FOUNDATION_PATH, "saved_model.pb")):
        print("Derm Foundation model not found locally. Downloading from Google Drive...")
        success = download_from_gdrive(GDRIVE_DERM_FOUNDATION_ID, DERM_FOUNDATION_PATH, is_folder=True)
        if not success:
            print("WARNING: Failed to download Derm Foundation model!")
    
    # Load Derm Foundation model
    local_model_paths = [
        DERM_FOUNDATION_PATH,
        "./derm_foundation/",
        "./",
        "./saved_model/",
        "./model/"
    ]
    
    for model_path in local_model_paths:
        saved_model_pb = os.path.join(model_path, "saved_model.pb")
        if os.path.exists(saved_model_pb):
            try:
                derm_model = tf.saved_model.load(model_path)
                print(f"Derm-Foundation model loaded from: {model_path}")
                break
            except Exception as e:
                print(f"Failed to load from {model_path}: {str(e)[:100]}")
                continue
    
    # Load EASI model (keep this local in your repo)
    model_path = './trained_model/easi_severity_model_derm_foundation_individual.pkl'
    if os.path.exists(model_path):
        easi_model = DermFoundationNeuralNetwork()
        success = easi_model.load_model(model_path)
        if success:
            print(f"EASI model loaded from: {model_path}")
        else:
            print(f"Failed to load EASI model")
            easi_model = None
    
    if derm_model is None or easi_model is None:
        print("WARNING: Some models failed to load!")


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "EASI Severity Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "conditions": "/conditions"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok" if (derm_model is not None and easi_model is not None) else "degraded",
        "models_loaded": {
            "derm_foundation": derm_model is not None,
            "easi_model": easi_model is not None
        },
        "available_conditions": len(easi_model.mlb.classes_) if easi_model else 0
    }


@app.get("/conditions", response_model=Dict[str, List[str]])
async def get_conditions():
    """Get list of available conditions"""
    if easi_model is None:
        raise HTTPException(status_code=503, detail="EASI model not loaded")
    
    return {
        "conditions": easi_model.mlb.classes_.tolist()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_easi(
    file: UploadFile = File(..., description="Skin image file (JPG, JPEG, PNG)")
):
    """
    Predict EASI scores from uploaded skin image.
    
    - **file**: Image file (JPG, JPEG, PNG)
    - Returns: EASI scores, component breakdown, and condition predictions
    """
    
    # Validate models loaded
    if derm_model is None or easi_model is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Check server logs."
        )
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPG, JPEG, PNG)"
        )
    
    try:
        # Read and process image
        image_bytes = await file.read()
        original_image = Image.open(BytesIO(image_bytes)).convert('RGB')
        original_size = original_image.size
        
        # Process to 448x448
        cropped_img = smart_crop_to_square(original_image)
        processed_img = cropped_img.resize((448, 448), Image.Resampling.LANCZOS)
        
        # Generate embedding
        embedding = generate_derm_foundation_embedding(derm_model, processed_img)
        
        # Make prediction
        predictions = easi_model.predict(embedding)
        
        if predictions is None:
            raise HTTPException(status_code=500, detail="Prediction failed")
        
        # Calculate EASI scores
        easi_results, total_easi = calculate_easi_scores(predictions)
        severity = get_severity_interpretation(total_easi)
        
        # Format predicted conditions
        predicted_conditions = []
        for i, condition in enumerate(predictions['dermatologist_skin_condition_on_label_name']):
            prob = predictions['all_condition_probabilities'][condition]
            conf = predictions['dermatologist_skin_condition_confidence'][i]
            weight = predictions['weighted_skin_condition_label'][condition]
            
            # Find EASI category
            easi_category = None
            easi_contribution = 0
            for cat_key, cat_info in easi_results.items():
                for contrib in cat_info['contributing_conditions']:
                    if contrib['condition'] == condition:
                        easi_category = cat_info['name']
                        easi_contribution = contrib['individual_score']
                        break
            
            predicted_conditions.append(ConditionPrediction(
                condition=condition,
                probability=float(prob),
                confidence=float(conf),
                weight=float(weight),
                easi_category=easi_category,
                easi_contribution=easi_contribution
            ))
        
        # Summary statistics
        summary_stats = {
            "total_conditions": len(predicted_conditions),
            "average_confidence": float(np.mean(predictions['dermatologist_skin_condition_confidence'])) if predicted_conditions else 0.0,
            "average_weight": float(np.mean(list(predictions['weighted_skin_condition_label'].values()))) if predicted_conditions else 0.0,
            "total_weight": float(sum(predictions['weighted_skin_condition_label'].values()))
        }
        
        # Format EASI components
        easi_components_formatted = {
            component: EASIComponent(
                name=result['name'],
                score=result['score'],
                contributing_conditions=result['contributing_conditions']
            )
            for component, result in easi_results.items()
        }
        
        return PredictionResponse(
            success=True,
            total_easi_score=total_easi,
            severity_interpretation=severity,
            easi_components=easi_components_formatted,
            predicted_conditions=predicted_conditions,
            summary_statistics=summary_stats,
            image_info={
                "original_size": f"{original_size[0]}x{original_size[1]}",
                "processed_size": "448x448",
                "filename": file.filename
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc)
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)