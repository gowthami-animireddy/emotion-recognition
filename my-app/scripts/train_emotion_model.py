"""
Training script for multi-modal emotion recognition model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
from pathlib import Path
from emotion_model import MultiModalEmotionFusion, EMOTIONS
from data_loader import get_dataloader

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        # Get inputs
        text = batch.get("text")
        audio = batch.get("audio")
        video = batch.get("video")
        labels = batch["label"].to(device)
        
        # Move to device
        if audio is not None:
            audio = audio.to(device)
        if video is not None:
            video = video.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits, _ = model(text=text, audio=audio, video=video)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            text = batch.get("text")
            audio = batch.get("audio")
            video = batch.get("video")
            labels = batch["label"].to(device)
            
            if audio is not None:
                audio = audio.to(device)
            if video is not None:
                video = video.to(device)
            
            logits, _ = model(text=text, audio=audio, video=video)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train emotion recognition model')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--modalities', nargs='+', default=['text', 'audio', 'video'], 
                       help='Modalities to use')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', 
                       help='Checkpoint directory')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    writer = SummaryWriter()
    
    # Model
    model = MultiModalEmotionFusion(num_emotions=len(EMOTIONS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                     patience=5, verbose=True)
    
    # Data
    train_loader = get_dataloader(args.data_dir, modalities=args.modalities, 
                                 split='train', batch_size=args.batch_size)
    val_loader = get_dataloader(args.data_dir, modalities=args.modalities, 
                               split='val', batch_size=args.batch_size)
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{args.epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / 'best_model.pt')
            print(f'Saved best model with val_loss: {val_loss:.4f}')
        
        scheduler.step(val_loss)
    
    writer.close()
    print("Training complete!")


if __name__ == "__main__":
    main()
