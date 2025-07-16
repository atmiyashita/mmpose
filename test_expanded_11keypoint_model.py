#!/usr/bin/env python3
"""
Test the new expanded 11-keypoint model and compare with previous models
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


def test_all_pose_models(image_path):
    """Test and compare all pose models"""
    
    print("🔍 Testing All Pose Models with Expanded Dataset")
    print("=" * 70)
    
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
    
    models_to_test = [
        {
            'name': 'Original 11-keypoint (118 annotations)',
            'config': 'configs/silkworm/td-hm_hrnet-w32_8xb64-210e_silkworm-keypoints-improved.py',
            'checkpoint': 'work_dirs/hrnet_silkworm_keypoints_improved/best_coco_AP_epoch_230.pth',
            'keypoints': 11
        },
        {
            'name': 'Expanded 11-keypoint (424 annotations)',
            'config': 'configs/silkworm/td-hm_hrnet-w32_8xb64-210e_silkworm-11keypoints-expanded.py',
            'checkpoint': 'work_dirs/hrnet_silkworm_11keypoints_expanded/best_coco_AP_epoch_200.pth',
            'keypoints': 11
        },
        {
            'name': '4-keypoint (95 annotations)',
            'config': 'configs/silkworm/td-hm_hrnet-w32_8xb64-210e_silkworm-4keypoints.py',
            'checkpoint': 'work_dirs/hrnet_silkworm_4keypoints/best_coco_AP_epoch_130.pth',
            'keypoints': 4
        }
    ]
    
    results = []
    
    for model_info in models_to_test:
        print(f"\n🔄 Testing {model_info['name']}...")
        try:
            pose_model = init_model(
                model_info['config'],
                model_info['checkpoint'],
                device=device)
            
            pose_results = inference_topdown(pose_model, image, [bbox])
            keypoint_scores = pose_results[0].pred_instances.keypoint_scores[0]
            keypoints = pose_results[0].pred_instances.keypoints[0]
            
            # Count good keypoints (>0.3 confidence)
            good_keypoints = len([score for score in keypoint_scores if score > 0.3])
            avg_conf = np.mean(keypoint_scores)
            max_conf = max(keypoint_scores)
            
            results.append({
                'name': model_info['name'],
                'total_keypoints': model_info['keypoints'],
                'good_keypoints': good_keypoints,
                'avg_conf': avg_conf,
                'max_conf': max_conf,
                'scores': keypoint_scores
            })
            
            print(f"✅ {model_info['name']} results:")
            print(f"   Good keypoints (>0.3): {good_keypoints}/{model_info['keypoints']}")
            print(f"   Average confidence: {avg_conf:.3f}")
            print(f"   Max confidence: {max_conf:.3f}")
            
            # Show individual keypoint scores for detailed analysis
            if model_info['keypoints'] == 11:
                keypoint_names = ['H1', 'T1', 'T2', 'T3', 'A2', 'A3', 'A4', 'A5', 'A6', 'A8', 'A9']
            else:
                keypoint_names = ['H1', 'T1', 'A2', 'A9']
            
            print(f"   Individual scores:")
            for i, (name, score) in enumerate(zip(keypoint_names, keypoint_scores)):
                status = "✓" if score > 0.3 else "✗"
                print(f"     {name}: {score:.3f} {status}")
        
        except Exception as e:
            print(f"❌ Error with {model_info['name']}: {e}")
            results.append({
                'name': model_info['name'],
                'total_keypoints': model_info['keypoints'],
                'good_keypoints': 0,
                'avg_conf': 0,
                'max_conf': 0,
                'scores': []
            })
    
    # Comparison summary
    print(f"\n📊 COMPREHENSIVE MODEL COMPARISON")
    print("=" * 80)
    print(f"{'Model':<35} {'Good/Total':<12} {'Reliability':<12} {'Avg Conf':<10} {'Max Conf':<10}")
    print("-" * 80)
    
    for result in results:
        reliability = result['good_keypoints'] / result['total_keypoints'] if result['total_keypoints'] > 0 else 0
        print(f"{result['name']:<35} {result['good_keypoints']}/{result['total_keypoints']:<11} {reliability:.1%}{'':>7} {result['avg_conf']:<10.3f} {result['max_conf']:<10.3f}")
    
    print(f"\n🎯 ANALYSIS:")
    
    # Find best model by reliability
    best_reliability = max(results, key=lambda x: x['good_keypoints']/x['total_keypoints'] if x['total_keypoints'] > 0 else 0)
    print(f"✅ Best reliability: {best_reliability['name']}")
    
    # Find best model by average confidence
    best_avg_conf = max(results, key=lambda x: x['avg_conf'])
    print(f"✅ Best average confidence: {best_avg_conf['name']} ({best_avg_conf['avg_conf']:.3f})")
    
    # Find best model by max confidence
    best_max_conf = max(results, key=lambda x: x['max_conf'])
    print(f"✅ Best max confidence: {best_max_conf['name']} ({best_max_conf['max_conf']:.3f})")
    
    print(f"\n🏆 RECOMMENDATION:")
    
    # Compare expanded 11-keypoint vs original 11-keypoint
    expanded_11 = next((r for r in results if 'Expanded 11-keypoint' in r['name']), None)
    original_11 = next((r for r in results if 'Original 11-keypoint' in r['name']), None)
    keypoint_4 = next((r for r in results if '4-keypoint' in r['name']), None)
    
    if expanded_11 and original_11:
        expanded_rel = expanded_11['good_keypoints'] / expanded_11['total_keypoints']
        original_rel = original_11['good_keypoints'] / original_11['total_keypoints']
        
        if expanded_rel > original_rel:
            print(f"🎉 EXPANDED 11-keypoint model shows improvement!")
            print(f"   Reliability: {expanded_rel:.1%} vs {original_rel:.1%} (+{(expanded_rel-original_rel)*100:.1f}%)")
            print(f"   Avg confidence: {expanded_11['avg_conf']:.3f} vs {original_11['avg_conf']:.3f}")
        
        if keypoint_4:
            keypoint_4_rel = keypoint_4['good_keypoints'] / keypoint_4['total_keypoints']
            print(f"\n📈 Model comparison:")
            print(f"   4-keypoint: {keypoint_4_rel:.1%} reliability, {keypoint_4['avg_conf']:.3f} avg conf")
            print(f"   11-keypoint (expanded): {expanded_rel:.1%} reliability, {expanded_11['avg_conf']:.3f} avg conf")
            
            if expanded_rel >= keypoint_4_rel and expanded_11['avg_conf'] >= keypoint_4['avg_conf']:
                print(f"🏆 RECOMMENDATION: Use EXPANDED 11-keypoint model!")
                print(f"   ✅ Better or equal reliability")
                print(f"   ✅ More detailed pose information (11 vs 4 keypoints)")
            elif keypoint_4_rel > expanded_rel:
                print(f"🏆 RECOMMENDATION: Use 4-keypoint model for reliability")
                print(f"   ✅ Higher reliability ({keypoint_4_rel:.1%} vs {expanded_rel:.1%})")
            else:
                print(f"🤔 Both models have trade-offs - choose based on your needs")


def main():
    parser = argparse.ArgumentParser(description='Test Expanded 11-Keypoint Model')
    parser.add_argument('image', help='Input image path')
    
    args = parser.parse_args()
    
    test_all_pose_models(args.image)


if __name__ == '__main__':
    main()
