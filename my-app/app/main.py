"""
FastAPI application for emotion recognition inference
This provides REST API endpoints for model predictions
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import librosa
import cv2
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Emotion Recognition API",
    description="Multi-modal emotion recognition service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


class TextPredictionRequest(BaseModel):
    text: str


class AudioPredictionRequest(BaseModel):
    audio_features: list[float]


class VideoPredictionRequest(BaseModel):
    video_features: list[float]


class PredictionResponse(BaseModel):
    primary_emotion: str
    confidence: float
    all_emotions: dict[str, float]
    attention_weights: dict[str, float]


@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model
    try:
        model_path = Path("./models/emotion_model.pt")
        if model_path.exists():
            # Import model class
            from scripts.emotion_model import MultiModalEmotionFusion
            model = MultiModalEmotionFusion(num_emotions=7)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            logger.info("Model loaded successfully")
        else:
            logger.warning(f"Model file not found at {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "emotion-recognition-api",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_loaded": model is not None
    }


@app.post("/predict/text")
async def predict_text(request: TextPredictionRequest) -> PredictionResponse:
    """Predict emotion from text"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        with torch.no_grad():
            logits, attention = model(text=request.text)
            probs = torch.softmax(logits, dim=1)
            
            top_prob, top_idx = torch.max(probs, 1)
            
            return PredictionResponse(
                primary_emotion=EMOTIONS[top_idx.item()],
                confidence=float(top_prob.item()),
                all_emotions={
                    emotion: float(prob)
                    for emotion, prob in zip(EMOTIONS, probs[0])
                },
                attention_weights={
                    "text": float(attention[0][0]),
                    "audio": float(attention[0][1]),
                    "video": float(attention[0][2])
                }
            )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/predict/audio")
async def predict_audio(audio_file: UploadFile = File(...)) -> PredictionResponse:
    """Predict emotion from audio file"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Save and process audio
        content = await audio_file.read()
        audio_path = f"/tmp/{audio_file.filename}"
        
        with open(audio_path, "wb") as f:
            f.write(content)
        
        # Extract features
        y, sr = librosa.load(audio_path, sr=22050)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        audio_tensor = torch.from_numpy(mel_spec_db.T).float().unsqueeze(0)
        
        with torch.no_grad():
            logits, attention = model(audio=audio_tensor)
            probs = torch.softmax(logits, dim=1)
            
            top_prob, top_idx = torch.max(probs, 1)
            
            return PredictionResponse(
                primary_emotion=EMOTIONS[top_idx.item()],
                confidence=float(top_prob.item()),
                all_emotions={
                    emotion: float(prob)
                    for emotion, prob in zip(EMOTIONS, probs[0])
                },
                attention_weights={
                    "text": float(attention[0][0]),
                    "audio": float(attention[0][1]),
                    "video": float(attention[0][2])
                }
            )
    except Exception as e:
        logger.error(f"Audio prediction error: {e}")
        raise HTTPException(status_code=500, detail="Audio prediction failed")


@app.post("/predict/video")
async def predict_video(video_file: UploadFile = File(...)) -> PredictionResponse:
    """Predict emotion from video file"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Save video
        content = await video_file.read()
        video_path = f"/tmp/{video_file.filename}"
        
        with open(video_path, "wb") as f:
            f.write(content)
        
        # Extract frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // 30)
        
        frame_count = 0
        while cap.isOpened() and len(frames) < 30:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0
                frames.append(frame)
            frame_count += 1
        
        cap.release()
        
        # Pad frames if needed
        while len(frames) < 30:
            frames.append(np.zeros((224, 224, 3)))
        
        video_tensor = torch.from_numpy(np.array(frames[:30])).float().unsqueeze(0)
        
        with torch.no_grad():
            logits, attention = model(video=video_tensor)
            probs = torch.softmax(logits, dim=1)
            
            top_prob, top_idx = torch.max(probs, 1)
            
            return PredictionResponse(
                primary_emotion=EMOTIONS[top_idx.item()],
                confidence=float(top_prob.item()),
                all_emotions={
                    emotion: float(prob)
                    for emotion, prob in zip(EMOTIONS, probs[0])
                },
                attention_weights={
                    "text": float(attention[0][0]),
                    "audio": float(attention[0][1]),
                    "video": float(attention[0][2])
                }
            )
    except Exception as e:
        logger.error(f"Video prediction error: {e}")
        raise HTTPException(status_code=500, detail="Video prediction failed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
