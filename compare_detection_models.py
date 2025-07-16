#!/usr/bin/env python3
"""
Compare old vs new detection models
"""

import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_detection_model(model_path, num_classes, device='cuda:0'):
    """Load detection model with specified number of classes"""
    device = torch.device(device)
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    return model, device


def detect_silkworms(model, device, image, score_threshold=0.5):
    """Run detection on image"""
    if isinstance(image, np.ndarray):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
    else:
        pil_image = image
    
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(pil_image).to(device)
    
    with torch.no_grad():
        outputs = model([img_tensor])
    
    output = outputs[0]
    boxes = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    labels = output["labels"].cpu().numpy()
    
    # Filter by score and exclude background
    valid_mask = (scores >= score_threshold) & (labels != 0)
    
    return boxes[valid_mask], scores[valid_mask], labels[valid_mask]


def visualize_comparison(image_path, old_model_path, new_model_path, output_path):
    """Create side-by-side comparison"""
    
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load models
    print("Loading old model (4 classes)...")
    old_model, device = load_detection_model(old_model_path, 4)
    
    print("Loading new model (5 classes)...")
    new_model, _ = load_detection_model(new_model_path, 5)
    
    # Run detections
    print("Running old model detection...")
    old_boxes, old_scores, old_labels = detect_silkworms(old_model, device, image)
    
    print("Running new model detection...")
    new_boxes, new_scores, new_labels = detect_silkworms(new_model, device, image)
    
    # Class names
    old_class_names = {1: "kaiko_live", 2: "kaiko_pao1", 3: "kaiko_rhi"}
    new_class_names = {1: "kaiko_live", 2: "kaiko_pao1", 3: "kaiko_rhi", 4: "kaiko"}
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Old model results
    ax1.imshow(image_rgb)
    ax1.set_title(f"Old Model\n{len(old_boxes)} detections")
    
    for bbox, score, label in zip(old_boxes, old_scores, old_labels):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        
        color = 'green' if label == 1 else 'red' if label == 2 else 'orange'
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                               edgecolor=color, facecolor='none')
        ax1.add_patch(rect)
        
        class_name = old_class_names.get(label, f"class_{label}")
        ax1.text(x1, y1-5, f"{class_name}: {score:.2f}", 
                color=color, fontsize=10, weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax1.axis('off')
    
    # New model results
    ax2.imshow(image_rgb)
    ax2.set_title(f"New Model (Trained)\n{len(new_boxes)} detections")
    
    for bbox, score, label in zip(new_boxes, new_scores, new_labels):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        
        if label == 1:
            color = 'green'
        elif label == 2:
            color = 'red'
        elif label == 3:
            color = 'orange'
        elif label == 4:
            color = 'blue'
        else:
            color = 'gray'
            
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                               edgecolor=color, facecolor='none')
        ax2.add_patch(rect)
        
        class_name = new_class_names.get(label, f"class_{label}")
        ax2.text(x1, y1-5, f"{class_name}: {score:.2f}", 
                color=color, fontsize=10, weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print comparison results
    print(f"\n📊 Detection Comparison Results:")
    print(f"{'Model':<15} {'Detections':<12} {'Max Confidence':<15} {'Avg Confidence':<15}")
    print("-" * 60)
    
    old_max_conf = max(old_scores) if len(old_scores) > 0 else 0
    old_avg_conf = np.mean(old_scores) if len(old_scores) > 0 else 0
    print(f"{'Old Model':<15} {len(old_boxes):<12} {old_max_conf:<15.3f} {old_avg_conf:<15.3f}")
    
    new_max_conf = max(new_scores) if len(new_scores) > 0 else 0
    new_avg_conf = np.mean(new_scores) if len(new_scores) > 0 else 0
    print(f"{'New Model':<15} {len(new_boxes):<12} {new_max_conf:<15.3f} {new_avg_conf:<15.3f}")
    
    print(f"\n📈 Improvements:")
    if len(new_boxes) > 0 and len(old_boxes) > 0:
        conf_improvement = new_max_conf - old_max_conf
        print(f"  Confidence improvement: {conf_improvement:+.3f}")
    
    print(f"  New model detected class: {new_class_names.get(new_labels[0], 'unknown') if len(new_labels) > 0 else 'none'}")
    
    return old_boxes, old_scores, new_boxes, new_scores


def main():
    image_path = "test1.jpg"
    old_model_path = "models/detection/fasterrcnn_kaiko_2025-0226_pao1_Ra46105.pth"
    new_model_path = "work_dirs/detection_training_extend/final_model.pth"
    output_path = "detection_comparison.png"
    
    print("🔍 Comparing Detection Models")
    print("=" * 50)
    
    old_boxes, old_scores, new_boxes, new_scores = visualize_comparison(
        image_path, old_model_path, new_model_path, output_path
    )
    
    print(f"\n✅ Comparison saved to: {output_path}")


if __name__ == '__main__':
    main()
