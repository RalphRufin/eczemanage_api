import os
import pickle
import tensorflow as tf
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DermFoundationModel:
    """Wrapper for Google's Derm Foundation model"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
    
    def load(self) -> bool:
        """Load the Derm Foundation SavedModel"""
        try:
            saved_model_pb = os.path.join(self.model_path, "saved_model.pb")
            
            if not os.path.exists(saved_model_pb):
                logger.error(f"Model file not found at {saved_model_pb}")
                return False
            
            self.model = tf.saved_model.load(self.model_path)
            logger.info(f"Derm Foundation model loaded from {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Derm Foundation model: {e}")
            return False
    
    def get_inference_function(self):
        """Get the model's inference signature"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return self.model.signatures["serving_default"]


class EASIModel:
    """Wrapper for EASI severity prediction model"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.mlb = None
        self.embedding_scaler = None
        self.confidence_scaler = None
        self.weighted_scaler = None
    
    def load(self) -> bool:
        """Load the EASI model and preprocessors"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"EASI model not found at {self.model_path}")
                return False
            
            logger.info(f"Loading pickle from {self.model_path}")
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            logger.info("Pickle loaded successfully")
            
            # Load preprocessing components
            self.mlb = model_data['mlb']
            self.embedding_scaler = model_data['embedding_scaler']
            self.confidence_scaler = model_data['confidence_scaler']
            self.weighted_scaler = model_data['weighted_scaler']
            
            logger.info("Preprocessors loaded")
            
            # Load Keras model
            keras_model_path = model_data['keras_model_path']
            logger.info(f"Keras model path: {keras_model_path}")
            
            if not os.path.exists(keras_model_path):
                logger.error(f"Keras model not found at {keras_model_path}")
                logger.error(f"Current working directory: {os.getcwd()}")
                logger.error(f"Files in models/trained_model/: {os.listdir('./models/trained_model/')}")
                return False
            
            logger.info(f"Loading Keras model from {keras_model_path}")
            self.model = tf.keras.models.load_model(keras_model_path)
            
            logger.info(f"EASI model loaded successfully from {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading EASI model: {e}", exc_info=True)
            return False
    
    def predict(self, embedding):
        """Make predictions on a single embedding"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        import numpy as np
        
        if len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)
        
        # Scale embedding
        embedding_scaled = self.embedding_scaler.transform(embedding)
        
        # Make predictions
        predictions = self.model.predict(embedding_scaled, verbose=0)
        
        # Process outputs
        condition_probs = predictions['conditions'][0]
        individual_confidences = predictions['individual_confidences'][0]
        individual_weights = predictions['individual_weights'][0]
        
        # Threshold for predictions
        condition_threshold = 0.3
        predicted_indices = np.where(condition_probs > condition_threshold)[0]
        
        # Build results
        predicted_conditions = []
        predicted_confidences = []
        predicted_weights_dict = {}
        
        for idx in predicted_indices:
            condition_name = self.mlb.classes_[idx]
            condition_prob = float(condition_probs[idx])
            
            # Inverse transform individual outputs
            if individual_confidences[idx] > 0:
                confidence_orig = self.confidence_scaler.inverse_transform(
                    [[individual_confidences[idx]]]
                )[0, 0]
            else:
                confidence_orig = 0.0
            
            if individual_weights[idx] > 0:
                weight_orig = self.weighted_scaler.inverse_transform(
                    [[individual_weights[idx]]]
                )[0, 0]
            else:
                weight_orig = 0.0
            
            predicted_conditions.append(condition_name)
            predicted_confidences.append(max(0, confidence_orig))
            predicted_weights_dict[condition_name] = max(0, weight_orig)
        
        # All condition probabilities
        all_condition_probs = {}
        all_confidences = {}
        all_weights = {}
        
        for i, class_name in enumerate(self.mlb.classes_):
            all_condition_probs[class_name] = float(condition_probs[i])
            
            if individual_confidences[i] > 0:
                conf_orig = self.confidence_scaler.inverse_transform(
                    [[individual_confidences[i]]]
                )[0, 0]
                all_confidences[class_name] = max(0, conf_orig)
            else:
                all_confidences[class_name] = 0.0
            
            if individual_weights[i] > 0:
                weight_orig = self.weighted_scaler.inverse_transform(
                    [[individual_weights[i]]]
                )[0, 0]
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


class ModelManager:
    """Singleton manager for all models"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.derm_model = None
            cls._instance.easi_model = None
        return cls._instance
    
    def load_models(self, derm_path: str, easi_path: str) -> bool:
        """Load both models"""
        logger.info("Loading models...")
        
        # Load Derm Foundation model
        self.derm_model = DermFoundationModel(derm_path)
        if not self.derm_model.load():
            return False
        
        # Load EASI model
        self.easi_model = EASIModel(easi_path)
        if not self.easi_model.load():
            return False
        
        logger.info("All models loaded successfully")
        return True
    
    def is_ready(self) -> bool:
        """Check if both models are loaded"""
        return (self.derm_model is not None and 
                self.derm_model.model is not None and
                self.easi_model is not None and 
                self.easi_model.model is not None)