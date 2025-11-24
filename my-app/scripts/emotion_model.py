"""
Hybrid CNN-RNN Emotion Recognition Model
Combines pre-trained models for text, audio, and video with a fusion layer
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import librosa
import cv2

# Text Model - BERT-based emotional embeddings
class TextEmotionModel(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_emotions=7):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.dense = nn.Linear(768, 512)
        self.batch_norm = nn.BatchNorm1d(512)
        self.emotion_head = nn.Linear(512, num_emotions)
        self.attention = nn.MultiheadAttention(768, num_heads=8, batch_first=True)
        
    def forward(self, text):
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.bert(**tokens)
        
        last_hidden = outputs.last_hidden_state
        attn_output, _ = self.attention(last_hidden, last_hidden, last_hidden)
        
        # Use mean pooling for robustness
        pooled = attn_output.mean(dim=1)
        
        hidden = self.dropout(F.relu(self.batch_norm(self.dense(pooled))))
        emotions = self.emotion_head(hidden)
        return emotions, hidden


# Audio Model - CNN-RNN for audio features
class AudioEmotionModel(nn.Module):
    def __init__(self, num_emotions=7):
        super().__init__()
        # Enhanced CNN for multi-scale feature extraction
        self.conv1 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1)
        
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.batch_norm3 = nn.BatchNorm1d(512)
        
        # Bidirectional LSTM for superior temporal context
        self.rnn = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, 
                          batch_first=True, bidirectional=True, dropout=0.4)
        
        self.attention = nn.MultiheadAttention(512, num_heads=4, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.4)
        self.dense = nn.Linear(512, 256)
        self.emotion_head = nn.Linear(256, num_emotions)
    
    def forward(self, mel_spectrogram):
        x = mel_spectrogram.transpose(1, 2)
        
        x = self.batch_norm1(F.relu(self.conv1(x)))
        x = self.batch_norm2(F.relu(self.conv2(x)))
        x = self.batch_norm3(F.relu(self.conv3(x)))
        x = x.transpose(1, 2)
        
        # Temporal modeling with attention
        rnn_out, (h_n, c_n) = self.rnn(x)
        attn_out, _ = self.attention(rnn_out, rnn_out, rnn_out)
        x = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=1)
        
        hidden = self.dropout(F.relu(self.dense(x)))
        emotions = self.emotion_head(hidden)
        return emotions, hidden


# Video Model - Frame-based CNN + temporal RNN
class VideoEmotionModel(nn.Module):
    def __init__(self, num_emotions=7):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Bidirectional GRU for temporal sequences
        self.rnn = nn.GRU(input_size=256, hidden_size=128, num_layers=2,
                         batch_first=True, bidirectional=True, dropout=0.3)
        self.dense = nn.Linear(256, 256)
        self.emotion_head = nn.Linear(256, num_emotions)
    
    def forward(self, frames):
        batch_size, num_frames = frames.shape[0], frames.shape[1]
        
        frame_features = []
        for i in range(num_frames):
            frame = frames[:, i, :, :, :]
            x = self.pool(F.relu(self.bn1(self.conv1(frame))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = self.adaptive_pool(x).view(x.size(0), -1)
            frame_features.append(x)
        
        frame_features = torch.stack(frame_features, dim=1)
        rnn_out, _ = self.rnn(frame_features)
        x = rnn_out[:, -1, :]
        
        hidden = F.relu(self.dense(x))
        emotions = self.emotion_head(hidden)
        return emotions, hidden


# Fusion Model - Combines all modalities
class MultiModalEmotionFusion(nn.Module):
    def __init__(self, num_emotions=7):
        super().__init__()
        self.text_model = TextEmotionModel(num_emotions=num_emotions)
        self.audio_model = AudioEmotionModel(num_emotions=num_emotions)
        self.video_model = VideoEmotionModel(num_emotions=num_emotions)
        
        # Fusion layer
        self.fusion_fc1 = nn.Linear(768, 512)
        self.fusion_fc2 = nn.Linear(512, 256)
        self.batch_norm = nn.BatchNorm1d(256)
        self.final_head = nn.Linear(256, num_emotions)
        self.dropout = nn.Dropout(0.3)
        
        # Attention weights for modality fusion
        self.attention_gate = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Softmax(dim=1)
        )
    
    def forward(self, text=None, audio=None, video=None):
        embeddings = []
        
        if text is not None:
            text_emotions, text_embedding = self.text_model(text)
            embeddings.append(text_embedding)
        
        if audio is not None:
            audio_emotions, audio_embedding = self.audio_model(audio)
            embeddings.append(audio_embedding)
        
        if video is not None:
            video_emotions, video_embedding = self.video_model(video)
            embeddings.append(video_embedding)
        
        # Concatenate embeddings
        combined = torch.cat(embeddings, dim=1)
        
        # Attention-weighted fusion
        attention_weights = self.attention_gate(combined)
        
        # Fused representation
        fused = self.dropout(F.relu(self.fusion_fc1(combined)))
        fused = self.dropout(F.relu(self.batch_norm(self.fusion_fc2(fused))))
        final_emotions = self.final_head(fused)
        
        return final_emotions, attention_weights


# Emotion classes
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def extract_audio_features(audio_path, sr=22050, n_mels=64):
    """Extract mel-spectrogram features from audio"""
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db.T  # (time_steps, n_mels)
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return None


def extract_video_frames(video_path, num_frames=30):
    """Extract frames from video"""
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // num_frames)
        
        while cap.isOpened() and len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0  # Normalize
                frames.append(frame)
            frame_count += 1
        
        cap.release()
        return np.array(frames)  # (num_frames, 224, 224, 3)
    except Exception as e:
        print(f"Error extracting video frames: {e}")
        return None


def predict_emotion(model, text=None, audio_path=None, video_path=None):
    """Predict emotion from multi-modal input"""
    model.eval()
    
    with torch.no_grad():
        # Prepare inputs
        audio_features = None
        video_frames = None
        
        if audio_path:
            audio_features = extract_audio_features(audio_path)
            if audio_features is not None:
                audio_features = torch.from_numpy(audio_features).float().unsqueeze(0)
        
        if video_path:
            video_frames = extract_video_frames(video_path)
            if video_frames is not None:
                video_frames = torch.from_numpy(video_frames).float().unsqueeze(0)
        
        # Forward pass
        logits, attention = model(text=text, audio=audio_features, video=video_frames)
        probabilities = F.softmax(logits, dim=1)
        
        # Get top emotions
        top_probs, top_indices = torch.topk(probabilities, k=3)
        
        results = {
            "primary_emotion": EMOTIONS[top_indices[0][0]],
            "confidence": float(top_probs[0][0]),
            "all_emotions": {EMOTIONS[i]: float(prob) for i, prob in zip(top_indices[0], top_probs[0])},
            "attention_weights": {
                "text": float(attention[0][0]) if text else 0,
                "audio": float(attention[0][1]) if audio_path else 0,
                "video": float(attention[0][2]) if video_path else 0
            }
        }
        
        return results


if __name__ == "__main__":
    # Example usage
    model = MultiModalEmotionFusion(num_emotions=7)
    
    # Test with text
    sample_text = "I am so happy today!"
    result = predict_emotion(model, text=sample_text)
    print(f"Text emotion prediction: {result}")
