"""
FastAPI inference server for hybrid CNN-RNN emotion recognition model
Loads trained model and serves predictions via REST API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
import logging
from typing import Optional, Dict, List

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))
from emotion_model import MultiModalEmotionFusion, EMOTIONS, extract_audio_features, extract_video_frames

app = FastAPI(title="Emotion Recognition Inference")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model
model = None
device = None

class TextInput(BaseModel):
    text: str

class AudioInput(BaseModel):
    audio_path: str

class VideoInput(BaseModel):
    video_path: str

class MultiModalInput(BaseModel):
    text: Optional[str] = None
    audio_path: Optional[str] = None
    video_path: Optional[str] = None

class PredictionResponse(BaseModel):
    primary_emotion: str
    confidence: float
    all_emotions: Dict[str, float]
    attention_weights: Dict[str, float]

@app.on_event("startup")
async def load_model():
    """Load model on server startup"""
    global model, device
    device = torch.device('cpu')
    
    try:
        model = MultiModalEmotionFusion(num_emotions=len(EMOTIONS)).to(device)
        
        # Try to load checkpoint if it exists
        checkpoint_path = Path(__file__).parent / "checkpoints" / "best_model.pt"
        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
            logger.info("Model checkpoint loaded successfully")
        else:
            logger.warning(f"No checkpoint found at {checkpoint_path}, using untrained model")
        
        model.eval()
        logger.info("Model loaded and ready for inference")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.post("/predict/text", response_model=PredictionResponse)
async def predict_text(input_data: TextInput):
    """Predict emotion from text"""
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        with torch.no_grad():
            logits, attention = model(text=input_data.text)
            probabilities = F.softmax(logits, dim=1)[0]
            
            top_prob, top_idx = torch.max(probabilities, dim=0)
            
            return PredictionResponse(
                primary_emotion=EMOTIONS[top_idx.item()],
                confidence=float(top_prob.item()),
                all_emotions={emotion: float(prob) for emotion, prob in zip(EMOTIONS, probabilities)},
                attention_weights={"text": 1.0, "audio": 0.0, "video": 0.0}
            )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/audio", response_model=PredictionResponse)
async def predict_audio(input_data: AudioInput):
    """Predict emotion from audio"""
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        audio_features = extract_audio_features(input_data.audio_path)
        if audio_features is None:
            raise HTTPException(status_code=400, detail="Failed to extract audio features")
        
        audio_tensor = torch.from_numpy(audio_features).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, attention = model(audio=audio_tensor)
            probabilities = F.softmax(logits, dim=1)[0]
            
            top_prob, top_idx = torch.max(probabilities, dim=0)
            
            return PredictionResponse(
                primary_emotion=EMOTIONS[top_idx.item()],
                confidence=float(top_prob.item()),
                all_emotions={emotion: float(prob) for emotion, prob in zip(EMOTIONS, probabilities)},
                attention_weights={"text": 0.0, "audio": 1.0, "video": 0.0}
            )
    except Exception as e:
        logger.error(f"Audio prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/video", response_model=PredictionResponse)
async def predict_video(input_data: VideoInput):
    """Predict emotion from video"""
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        video_frames = extract_video_frames(input_data.video_path)
        if video_frames is None:
            raise HTTPException(status_code=400, detail="Failed to extract video frames")
        
        video_tensor = torch.from_numpy(video_frames).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, attention = model(video=video_tensor)
            probabilities = F.softmax(logits, dim=1)[0]
            
            top_prob, top_idx = torch.max(probabilities, dim=0)
            
            return PredictionResponse(
                primary_emotion=EMOTIONS[top_idx.item()],
                confidence=float(top_prob.item()),
                all_emotions={emotion: float(prob) for emotion, prob in zip(EMOTIONS, probabilities)},
                attention_weights={"text": 0.0, "audio": 0.0, "video": 1.0}
            )
    except Exception as e:
        logger.error(f"Video prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict_multimodal(input_data: MultiModalInput):
    """Predict emotion from multi-modal input (text, audio, video)"""
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not any([input_data.text, input_data.audio_path, input_data.video_path]):
        raise HTTPException(status_code=400, detail="At least one modality must be provided")
    
    try:
        audio_tensor = None
        video_tensor = None
        attention = None
        
        if input_data.audio_path:
            audio_features = extract_audio_features(input_data.audio_path)
            if audio_features is not None:
                audio_tensor = torch.from_numpy(audio_features).float().unsqueeze(0).to(device)
        
        if input_data.video_path:
            video_frames = extract_video_frames(input_data.video_path)
            if video_frames is not None:
                video_tensor = torch.from_numpy(video_frames).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, attention = model(text=input_data.text, audio=audio_tensor, video=video_tensor)
            probabilities = F.softmax(logits, dim=1)[0]
            
            top_prob, top_idx = torch.max(probabilities, dim=0)
            
            attention_dict = {
                "text": float(attention[0][0].item()) if input_data.text else 0.0,
                "audio": float(attention[0][1].item()) if input_data.audio_path else 0.0,
                "video": float(attention[0][2].item()) if input_data.video_path else 0.0
            }
            
            return PredictionResponse(
                primary_emotion=EMOTIONS[top_idx.item()],
                confidence=float(top_prob.item()),
                all_emotions={emotion: float(prob) for emotion, prob in zip(EMOTIONS, probabilities)},
                attention_weights=attention_dict
            )
    except Exception as e:
        logger.error(f"Multi-modal prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
