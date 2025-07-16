#!/usr/bin/env python3
"""
Test 4-keypoint pose model and compare with 11-keypoint model
"""

import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules
import argparse


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


def test_pose_models(image_path):
    """Test and compare 4-keypoint vs 11-keypoint models"""
    
    print("🔍 Testing 4-Keypoint vs 11-Keypoint Models")
    print("=" * 60)
    
    # Load image
    image = cv2.imread(image_path)
    print(f"📷 Testing on: {image_path}")
    
    # Register MMPose modules
    register_all_modules()
    
    # Load detection model
    print(f"\n🔄 Loading detection model...")
    det_model, device = load_detection_model(
        'work_dirs/detection_training_extend/final_model.pth', 5)
    boxes, scores, labels = detect_silkworms(det_model, device, image, 0.3)
    
    if len(boxes) == 0:
        print("❌ No silkworms detected!")
        return
    
    print(f"✅ Detection: {len(boxes)} silkworms found")
    print(f"   Best confidence: {max(scores):.3f}")
    
    # Use first detection for pose estimation
    bbox = boxes[0]
    print(f"   Using bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
    
    # Test 11-keypoint model
    print(f"\n🔄 Testing 11-keypoint model...")
    try:
        pose_model_11 = init_model(
            'configs/silkworm/td-hm_hrnet-w32_8xb64-210e_silkworm-256x192.py',
            'work_dirs/hrnet_silkworm_keypoints_improved/best_coco_AP_epoch_230.pth',
            device=device)
        
        pose_results_11 = inference_topdown(pose_model_11, image, [bbox])
        keypoint_scores_11 = pose_results_11[0].pred_instances.keypoint_scores[0]
        keypoints_11 = pose_results_11[0].pred_instances.keypoints[0]
        
        # Count good keypoints (>0.3 confidence)
        good_keypoints_11 = len([score for score in keypoint_scores_11 if score > 0.3])
        avg_conf_11 = np.mean(keypoint_scores_11)
        max_conf_11 = max(keypoint_scores_11)
        
        print(f"✅ 11-keypoint model results:")
        print(f"   Good keypoints (>0.3): {good_keypoints_11}/11")
        print(f"   Average confidence: {avg_conf_11:.3f}")
        print(f"   Max confidence: {max_conf_11:.3f}")
        
        # Show individual keypoint scores
        keypoint_names_11 = ['H1', 'T1', 'T2', 'T3', 'A2', 'A3', 'A4', 'A5', 'A6', 'A8', 'A9']
        print(f"   Individual scores:")
        for i, (name, score) in enumerate(zip(keypoint_names_11, keypoint_scores_11)):
            status = "✓" if score > 0.3 else "✗"
            print(f"     {name}: {score:.3f} {status}")
        
    except Exception as e:
        print(f"❌ Error with 11-keypoint model: {e}")
        good_keypoints_11, avg_conf_11, max_conf_11 = 0, 0, 0
    
    # Test 4-keypoint model
    print(f"\n🔄 Testing 4-keypoint model...")
    try:
        pose_model_4 = init_model(
            'configs/silkworm/td-hm_hrnet-w32_8xb64-210e_silkworm-4keypoints.py',
            'work_dirs/hrnet_silkworm_4keypoints/best_coco_AP_epoch_130.pth',
            device=device)
        
        pose_results_4 = inference_topdown(pose_model_4, image, [bbox])
        keypoint_scores_4 = pose_results_4[0].pred_instances.keypoint_scores[0]
        keypoints_4 = pose_results_4[0].pred_instances.keypoints[0]
        
        # Count good keypoints (>0.3 confidence)
        good_keypoints_4 = len([score for score in keypoint_scores_4 if score > 0.3])
        avg_conf_4 = np.mean(keypoint_scores_4)
        max_conf_4 = max(keypoint_scores_4)
        
        print(f"✅ 4-keypoint model results:")
        print(f"   Good keypoints (>0.3): {good_keypoints_4}/4")
        print(f"   Average confidence: {avg_conf_4:.3f}")
        print(f"   Max confidence: {max_conf_4:.3f}")
        
        # Show individual keypoint scores
        keypoint_names_4 = ['H1', 'T1', 'A2', 'A9']
        print(f"   Individual scores:")
        for i, (name, score) in enumerate(zip(keypoint_names_4, keypoint_scores_4)):
            status = "✓" if score > 0.3 else "✗"
            print(f"     {name}: {score:.3f} {status}")
        
    except Exception as e:
        print(f"❌ Error with 4-keypoint model: {e}")
        good_keypoints_4, avg_conf_4, max_conf_4 = 0, 0, 0
    
    # Comparison summary
    print(f"\n📊 MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<25} {'11-Keypoint':<15} {'4-Keypoint':<15} {'Winner':<15}")
    print("-" * 70)
    
    # Good keypoints
    good_ratio_11 = good_keypoints_11 / 11 if good_keypoints_11 > 0 else 0
    good_ratio_4 = good_keypoints_4 / 4 if good_keypoints_4 > 0 else 0
    winner_good = "4-Keypoint" if good_ratio_4 > good_ratio_11 else "11-Keypoint" if good_ratio_11 > good_ratio_4 else "Tie"
    print(f"{'Good Keypoints':<25} {good_keypoints_11}/11 ({good_ratio_11:.1%})<15 {good_keypoints_4}/4 ({good_ratio_4:.1%})<15 {winner_good:<15}")
    
    # Average confidence
    winner_avg = "4-Keypoint" if avg_conf_4 > avg_conf_11 else "11-Keypoint" if avg_conf_11 > avg_conf_4 else "Tie"
    print(f"{'Avg Confidence':<25} {avg_conf_11:<15.3f} {avg_conf_4:<15.3f} {winner_avg:<15}")
    
    # Max confidence
    winner_max = "4-Keypoint" if max_conf_4 > max_conf_11 else "11-Keypoint" if max_conf_11 > max_conf_4 else "Tie"
    print(f"{'Max Confidence':<25} {max_conf_11:<15.3f} {max_conf_4:<15.3f} {winner_max:<15}")
    
    print(f"\n🎯 ANALYSIS:")
    if good_ratio_4 > good_ratio_11:
        print(f"✅ 4-keypoint model has BETTER reliability ({good_ratio_4:.1%} vs {good_ratio_11:.1%})")
    elif good_ratio_4 == good_ratio_11:
        print(f"➡️ Both models have SAME reliability ({good_ratio_4:.1%})")
    else:
        print(f"⚠️ 11-keypoint model has better reliability ({good_ratio_11:.1%} vs {good_ratio_4:.1%})")
    
    if avg_conf_4 > avg_conf_11:
        print(f"✅ 4-keypoint model has HIGHER average confidence")
    elif avg_conf_4 == avg_conf_11:
        print(f"➡️ Both models have SAME average confidence")
    else:
        print(f"⚠️ 11-keypoint model has higher average confidence")
    
    print(f"\n🏆 RECOMMENDATION:")
    if good_ratio_4 >= good_ratio_11 and avg_conf_4 >= avg_conf_11:
        print(f"✅ Use 4-KEYPOINT model - Better reliability and confidence!")
    elif good_ratio_4 >= 0.75:  # 3/4 keypoints working
        print(f"✅ Use 4-KEYPOINT model - Good reliability with key landmarks!")
    else:
        print(f"🤔 Consider model improvements or different approach")


def main():
    parser = argparse.ArgumentParser(description='Test 4-Keypoint Model')
    parser.add_argument('image', help='Input image path')
    
    args = parser.parse_args()
    
    test_pose_models(args.image)


if __name__ == '__main__':
    main()
