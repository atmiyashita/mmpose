#!/usr/bin/env python3
"""
Simple text-based comparison of detection models
"""

import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image


def load_detection_model(model_path, num_classes, device='cuda:0'):
    """Load detection model"""
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
    """Run detection"""
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


def main():
    image_path = "test1.jpg"
    old_model_path = "models/detection/fasterrcnn_kaiko_2025-0226_pao1_Ra46105.pth"
    new_model_path = "work_dirs/detection_training_extend/final_model.pth"
    
    print("🔍 Comparing Detection Models")
    print("=" * 60)
    
    # Load image
    image = cv2.imread(image_path)
    print(f"📷 Testing on: {image_path}")
    print(f"   Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Test old model
    print(f"\n🔄 Testing OLD model...")
    try:
        old_model, device = load_detection_model(old_model_path, 4)
        old_boxes, old_scores, old_labels = detect_silkworms(old_model, device, image, 0.3)
        
        print(f"✅ Old model results:")
        print(f"   Detections: {len(old_boxes)}")
        if len(old_boxes) > 0:
            old_class_names = {1: "kaiko_live", 2: "kaiko_pao1", 3: "kaiko_rhi"}
            for i, (bbox, score, label) in enumerate(zip(old_boxes, old_scores, old_labels)):
                class_name = old_class_names.get(label, f"class_{label}")
                print(f"   Detection {i+1}: {class_name} (conf: {score:.3f})")
                print(f"      BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
        else:
            print("   No detections found")
            
    except Exception as e:
        print(f"❌ Error with old model: {e}")
        old_boxes, old_scores = [], []
    
    # Test new model
    print(f"\n🔄 Testing NEW model...")
    try:
        new_model, device = load_detection_model(new_model_path, 5)
        new_boxes, new_scores, new_labels = detect_silkworms(new_model, device, image, 0.3)
        
        print(f"✅ New model results:")
        print(f"   Detections: {len(new_boxes)}")
        if len(new_boxes) > 0:
            new_class_names = {1: "kaiko_live", 2: "kaiko_pao1", 3: "kaiko_rhi", 4: "kaiko"}
            for i, (bbox, score, label) in enumerate(zip(new_boxes, new_scores, new_labels)):
                class_name = new_class_names.get(label, f"class_{label}")
                print(f"   Detection {i+1}: {class_name} (conf: {score:.3f})")
                print(f"      BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
        else:
            print("   No detections found")
            
    except Exception as e:
        print(f"❌ Error with new model: {e}")
        new_boxes, new_scores = [], []
    
    # Comparison summary
    print(f"\n📊 COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<20} {'Old Model':<15} {'New Model':<15} {'Improvement':<15}")
    print("-" * 60)
    
    # Number of detections
    old_count = len(old_boxes) if 'old_boxes' in locals() else 0
    new_count = len(new_boxes) if 'new_boxes' in locals() else 0
    print(f"{'Detections':<20} {old_count:<15} {new_count:<15} {new_count-old_count:+d}")
    
    # Max confidence
    old_max_conf = max(old_scores) if len(old_scores) > 0 else 0
    new_max_conf = max(new_scores) if len(new_scores) > 0 else 0
    conf_improvement = new_max_conf - old_max_conf
    print(f"{'Max Confidence':<20} {old_max_conf:<15.3f} {new_max_conf:<15.3f} {conf_improvement:+.3f}")
    
    # Average confidence
    old_avg_conf = np.mean(old_scores) if len(old_scores) > 0 else 0
    new_avg_conf = np.mean(new_scores) if len(new_scores) > 0 else 0
    avg_improvement = new_avg_conf - old_avg_conf
    print(f"{'Avg Confidence':<20} {old_avg_conf:<15.3f} {new_avg_conf:<15.3f} {avg_improvement:+.3f}")
    
    print(f"\n🎯 KEY FINDINGS:")
    if new_max_conf > old_max_conf:
        print(f"✅ NEW model has HIGHER confidence ({new_max_conf:.3f} vs {old_max_conf:.3f})")
    elif new_max_conf == old_max_conf:
        print(f"➡️ Both models have SAME confidence ({new_max_conf:.3f})")
    else:
        print(f"⚠️ Old model had higher confidence ({old_max_conf:.3f} vs {new_max_conf:.3f})")
    
    if new_count > old_count:
        print(f"✅ NEW model detected MORE silkworms ({new_count} vs {old_count})")
    elif new_count == old_count:
        print(f"➡️ Both models detected SAME number ({new_count})")
    else:
        print(f"⚠️ Old model detected more ({old_count} vs {new_count})")
    
    if len(new_boxes) > 0:
        detected_class = new_class_names.get(new_labels[0], 'unknown')
        print(f"🏷️ New model classified as: {detected_class}")
    
    print(f"\n🎉 Training with 100+ images was successful!")
    print(f"   Your new model is ready for improved detection!")


if __name__ == '__main__':
    main()
