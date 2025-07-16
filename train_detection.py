#!/usr/bin/env python3
"""
Training script for Faster R-CNN silkworm detection
Fine-tunes from your existing pretrained model
"""

import os
import json
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt


class SilkwormDataset(Dataset):
    """Custom dataset for silkworm detection in COCO format"""
    
    def __init__(self, root_dir, annotation_file, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        
        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create image id to annotations mapping
        self.image_info = {img['id']: img for img in self.coco_data['images']}
        
        # Group annotations by image_id
        self.annotations_by_image = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.annotations_by_image:
                self.annotations_by_image[image_id] = []
            self.annotations_by_image[image_id].append(ann)
        
        # Get list of image IDs that have annotations
        self.image_ids = list(self.annotations_by_image.keys())
        
        print(f"Dataset loaded: {len(self.image_ids)} images with annotations")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.image_info[image_id]
        
        # Load image
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")
        
        # Get annotations for this image
        annotations = self.annotations_by_image[image_id]
        
        # Extract bounding boxes and labels
        boxes = []
        labels = []
        
        for ann in annotations:
            bbox = ann['bbox']  # [x, y, width, height]
            # Convert to [x1, y1, x2, y2]
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            boxes.append([x1, y1, x2, y2])
            
            # Map category_id to label (1-indexed for torchvision)
            # Your dataset has category_id=1 for "kaiko", keep it as 1
            labels.append(ann['category_id'])
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Calculate areas
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # Create target dict
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id]),
            "area": areas,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        if self.transforms:
            image = self.transforms(image)
        
        return image, target


def get_model(num_classes, pretrained_path=None):
    """Create Faster R-CNN model"""
    # Load pretrained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Replace the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Load your pretrained weights if provided
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
            print("✅ Pretrained weights loaded successfully!")
        except Exception as e:
            print(f"⚠️ Could not load pretrained weights: {e}")
            print("Starting from torchvision pretrained weights only")
    
    return model


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(data_loader)
    
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        
        # Update progress bar
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({
            'loss': f'{losses.item():.4f}',
            'avg_loss': f'{avg_loss:.4f}'
        })
    
    return total_loss / num_batches


def evaluate_model(model, data_loader, device):
    """Evaluate model on validation set"""
    model.train()  # Keep in training mode to get loss values
    total_loss = 0
    num_batches = len(data_loader)

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # In training mode, model returns loss dict
            loss_dict = model(images, targets)
            if isinstance(loss_dict, dict):
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
            else:
                # If model is in eval mode, it returns predictions, not losses
                # Switch to training mode temporarily
                model.train()
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
                model.eval()

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN for Silkworm Detection')
    parser.add_argument('--data-root', default='kaiko_labeled/images/default',
                       help='Root directory of images')
    parser.add_argument('--annotation-file',
                       default='work_dirs/detection_data/instances_consistent.json',
                       help='COCO annotation file (use consistent labels)')
    parser.add_argument('--pretrained-model',
                       default='models/detection/fasterrcnn_kaiko_2025-0226_pao1_Ra46105.pth',
                       help='Path to pretrained model weights')
    parser.add_argument('--output-dir', default='work_dirs/detection_training',
                       help='Output directory for trained models')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', default='cuda:0',
                       help='Device to use for training')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Fraction of data to use for validation')
    parser.add_argument('--num-classes', type=int, default=5,
                       help='Number of classes (including background). Use 4 for replace strategy, 5 for extend strategy')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load dataset
    full_dataset = SilkwormDataset(
        root_dir=args.data_root,
        annotation_file=args.annotation_file,
        transforms=train_transforms
    )
    
    # Split dataset into train and validation
    dataset_size = len(full_dataset)
    val_size = int(args.val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Create model with specified number of classes
    model = get_model(args.num_classes, args.pretrained_model)
    print(f"Model created with {args.num_classes} classes (including background)")
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=0.9, 
        weight_decay=0.0005
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.1
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch + 1)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = evaluate_model(model, val_loader, device)
        val_losses.append(val_loss)
        
        # Update learning rate
        lr_scheduler.step()
        
        print(f"Epoch {epoch + 1}/{args.epochs}: "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.output_dir, f'model_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"✅ Final model saved: {final_model_path}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(args.output_dir, 'training_curves.png')
    plt.savefig(plot_path)
    print(f"📊 Training curves saved: {plot_path}")
    
    print(f"\n🎉 Training completed!")
    print(f"📁 All outputs saved in: {args.output_dir}")


if __name__ == '__main__':
    main()
