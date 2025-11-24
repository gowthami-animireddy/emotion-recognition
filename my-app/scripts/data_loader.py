"""
Data loading and preprocessing for emotion recognition
Supports text, audio, and video modalities
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import librosa
import cv2
from transformers import AutoTokenizer

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}


class MultiModalEmotionDataset(Dataset):
    """Dataset for multi-modal emotion recognition"""
    
    def __init__(self, data_dir, modalities=["text"], split="train"):
        self.data_dir = Path(data_dir)
        self.modalities = modalities
        self.split = split
        self.samples = []
        
        # Load sample paths and labels
        self._load_samples()
        
        if "text" in modalities:
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    def _load_samples(self):
        """Load data samples from directory structure"""
        split_dir = self.data_dir / self.split
        for emotion_dir in split_dir.iterdir():
            if emotion_dir.is_dir() and emotion_dir.name in EMOTION_TO_IDX:
                for file in emotion_dir.iterdir():
                    self.samples.append({
                        "path": file,
                        "label": EMOTION_TO_IDX[emotion_dir.name],
                        "emotion": emotion_dir.name
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample["label"]
        file_path = sample["path"]
        
        data = {}
        
        # Load text
        if "text" in self.modalities and file_path.suffix == ".txt":
            with open(file_path, "r") as f:
                text = f.read()
            tokens = self.tokenizer(text, return_tensors="pt", padding=True, 
                                   truncation=True, max_length=128)
            data["text"] = text
            data["text_tokens"] = tokens
        
        # Load audio
        if "audio" in self.modalities and file_path.suffix in [".wav", ".mp3"]:
            y, sr = librosa.load(str(file_path), sr=22050)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            data["audio"] = torch.from_numpy(mel_spec_db.T).float()
        
        # Load video
        if "video" in self.modalities and file_path.suffix in [".mp4", ".avi"]:
            frames = self._extract_frames(str(file_path), num_frames=30)
            data["video"] = torch.from_numpy(frames).float()
        
        data["label"] = torch.tensor(label, dtype=torch.long)
        return data
    
    def _extract_frames(self, video_path, num_frames=30):
        """Extract frames from video file"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // num_frames)
        
        frame_count = 0
        while cap.isOpened() and len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame / 255.0
                frames.append(frame)
            frame_count += 1
        
        cap.release()
        
        # Pad with zeros if not enough frames
        while len(frames) < num_frames:
            frames.append(np.zeros((224, 224, 3)))
        
        return np.array(frames[:num_frames])


def get_dataloader(data_dir, modalities=["text"], split="train", batch_size=32):
    """Create dataloader for emotion recognition"""
    dataset = MultiModalEmotionDataset(data_dir, modalities=modalities, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"))
    return dataloader


if __name__ == "__main__":
    # Example usage
    dataset = MultiModalEmotionDataset("./data", modalities=["text", "audio"], split="train")
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
