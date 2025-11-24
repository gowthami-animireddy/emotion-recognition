"""
Data preprocessing utilities for emotion recognition datasets
Handles text, audio, and video data preparation
"""

import os
import json
import numpy as np
import librosa
import cv2
from pathlib import Path
import argparse


def prepare_text_data(input_file, output_dir):
    """Prepare text emotion data"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    for emotion, texts in data.items():
        emotion_dir = output_dir / emotion
        emotion_dir.mkdir(exist_ok=True)
        
        for idx, text in enumerate(texts):
            with open(emotion_dir / f"{idx}.txt", 'w') as f:
                f.write(text)
    
    print(f"Text data prepared in {output_dir}")


def prepare_audio_data(input_dir, output_dir, target_sr=22050):
    """Prepare audio emotion data"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for emotion_dir in input_dir.iterdir():
        if emotion_dir.is_dir():
            emotion = emotion_dir.name
            out_emotion_dir = output_dir / emotion
            out_emotion_dir.mkdir(exist_ok=True)
            
            for audio_file in emotion_dir.glob("*.wav"):
                try:
                    # Load and resample audio
                    y, sr = librosa.load(str(audio_file), sr=target_sr)
                    
                    # Save resampled audio
                    output_path = out_emotion_dir / audio_file.name
                    librosa.output.write_wav(str(output_path), y, sr)
                    print(f"Processed: {audio_file}")
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
    
    print(f"Audio data prepared in {output_dir}")


def prepare_video_data(input_dir, output_dir, target_fps=30):
    """Prepare video emotion data"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for emotion_dir in input_dir.iterdir():
        if emotion_dir.is_dir():
            emotion = emotion_dir.name
            out_emotion_dir = output_dir / emotion
            out_emotion_dir.mkdir(exist_ok=True)
            
            for video_file in emotion_dir.glob("*.mp4"):
                try:
                    cap = cv2.VideoCapture(str(video_file))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    # Only process if fps differs significantly
                    if abs(fps - target_fps) > 1:
                        output_path = out_emotion_dir / video_file.name
                        
                        # Re-encode video at target fps
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(str(output_path), fourcc, target_fps, 
                                            (int(cap.get(3)), int(cap.get(4))))
                        
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            out.write(frame)
                        
                        out.release()
                    else:
                        # Copy file as-is
                        import shutil
                        shutil.copy(video_file, out_emotion_dir / video_file.name)
                    
                    print(f"Processed: {video_file}")
                except Exception as e:
                    print(f"Error processing {video_file}: {e}")
                finally:
                    cap.release()
    
    print(f"Video data prepared in {output_dir}")


def split_dataset(data_dir, train_ratio=0.7, val_ratio=0.15):
    """Split dataset into train/val/test"""
    import random
    import shutil
    
    data_dir = Path(data_dir)
    splits = ['train', 'val', 'test']
    
    for split in splits:
        (data_dir / split).mkdir(exist_ok=True)
    
    # Collect all emotions
    emotions = [d.name for d in data_dir.iterdir() if d.is_dir() and d.name not in splits]
    
    for emotion in emotions:
        emotion_dir = data_dir / emotion
        files = list(emotion_dir.glob("*"))
        random.shuffle(files)
        
        train_idx = int(len(files) * train_ratio)
        val_idx = train_idx + int(len(files) * val_ratio)
        
        train_files = files[:train_idx]
        val_files = files[train_idx:val_idx]
        test_files = files[val_idx:]
        
        for split_name, split_files in [('train', train_files), ('val', val_files), 
                                        ('test', test_files)]:
            split_emotion_dir = data_dir / split_name / emotion
            split_emotion_dir.mkdir(parents=True, exist_ok=True)
            
            for file in split_files:
                shutil.copy(file, split_emotion_dir / file.name)
    
    print(f"Dataset split: Train {train_ratio*100}%, Val {val_ratio*100}%, Test {(1-train_ratio-val_ratio)*100}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess emotion recognition data')
    parser.add_argument('--modality', choices=['text', 'audio', 'video'], 
                       help='Data modality to preprocess')
    parser.add_argument('--input', required=True, help='Input file or directory')
    parser.add_argument('--output', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    if args.modality == 'text':
        prepare_text_data(args.input, args.output)
    elif args.modality == 'audio':
        prepare_audio_data(args.input, args.output)
    elif args.modality == 'video':
        prepare_video_data(args.input, args.output)
