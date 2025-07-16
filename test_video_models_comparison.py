#!/usr/bin/env python3
"""
Quick video test comparison of different model combinations
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
import time


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


def test_video_models(video_path, num_frames=100):
    """Test different model combinations on video frames"""
    
    print("🎬 Testing Model Combinations on Video")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Testing first {num_frames} frames")
    
    # Register MMPose modules
    register_all_modules()
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return
    
    # Model configurations to test
    model_configs = [
        {
            'name': 'Original Models',
            'detection': ('models/detection/fasterrcnn_kaiko_2025-0226_pao1_Ra46105.pth', 4),
            'pose': ('configs/silkworm/td-hm_hrnet-w32_8xb64-210e_silkworm-keypoints-improved.py',
                    'work_dirs/hrnet_silkworm_keypoints_improved/best_coco_AP_epoch_230.pth')
        },
        {
            'name': 'Improved Models (11-keypoint)',
            'detection': ('work_dirs/detection_training_extend/final_model.pth', 5),
            'pose': ('configs/silkworm/td-hm_hrnet-w32_8xb64-210e_silkworm-11keypoints-expanded.py',
                    'work_dirs/hrnet_silkworm_11keypoints_expanded/best_coco_AP_epoch_200.pth')
        },
        {
            'name': 'Improved Detection + 4-keypoint',
            'detection': ('work_dirs/detection_training_extend/final_model.pth', 5),
            'pose': ('configs/silkworm/td-hm_hrnet-w32_8xb64-210e_silkworm-4keypoints.py',
                    'work_dirs/hrnet_silkworm_4keypoints/best_coco_AP_epoch_130.pth')
        }
    ]
    
    results = {}
    
    for config in model_configs:
        print(f"\n🔄 Testing {config['name']}...")
        
        try:
            # Load models
            det_model, device = load_detection_model(config['detection'][0], config['detection'][1])
            pose_model = init_model(config['pose'][0], config['pose'][1], device=device)
            
            # Reset video to beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Test on frames
            total_detections = 0
            total_good_keypoints = 0
            total_confidence = 0
            frames_with_detections = 0
            processing_times = []
            
            for frame_idx in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                start_time = time.time()
                
                # Detection
                boxes, scores, labels = detect_silkworms(det_model, device, frame, 0.5)
                
                if len(boxes) > 0:
                    frames_with_detections += 1
                    total_detections += len(boxes)
                    
                    # Pose estimation on first detection
                    bbox = boxes[0]
                    pose_results = inference_topdown(pose_model, frame, [bbox])
                    keypoint_scores = pose_results[0].pred_instances.keypoint_scores[0]
                    
                    good_keypoints = len([score for score in keypoint_scores if score > 0.3])
                    total_good_keypoints += good_keypoints
                    total_confidence += np.mean(keypoint_scores)
                
                processing_times.append(time.time() - start_time)
            
            # Calculate metrics
            avg_detections_per_frame = total_detections / num_frames if num_frames > 0 else 0
            avg_good_keypoints = total_good_keypoints / frames_with_detections if frames_with_detections > 0 else 0
            avg_confidence = total_confidence / frames_with_detections if frames_with_detections > 0 else 0
            avg_processing_time = np.mean(processing_times)
            fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
            
            results[config['name']] = {
                'frames_with_detections': frames_with_detections,
                'detection_rate': frames_with_detections / num_frames,
                'avg_detections_per_frame': avg_detections_per_frame,
                'avg_good_keypoints': avg_good_keypoints,
                'avg_confidence': avg_confidence,
                'avg_processing_time': avg_processing_time,
                'fps': fps
            }
            
            print(f"✅ {config['name']} completed:")
            print(f"   Detection rate: {frames_with_detections}/{num_frames} ({frames_with_detections/num_frames:.1%})")
            print(f"   Avg detections/frame: {avg_detections_per_frame:.2f}")
            print(f"   Avg good keypoints: {avg_good_keypoints:.1f}")
            print(f"   Avg confidence: {avg_confidence:.3f}")
            print(f"   Processing speed: {fps:.1f} fps")
            
        except Exception as e:
            print(f"❌ Error with {config['name']}: {e}")
            results[config['name']] = None
    
    cap.release()
    
    # Comparison summary
    print(f"\n📊 VIDEO MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Model':<30} {'Detection Rate':<15} {'Avg Det/Frame':<15} {'Avg Keypts':<12} {'Avg Conf':<10} {'FPS':<8}")
    print("-" * 80)
    
    for name, result in results.items():
        if result:
            print(f"{name:<30} {result['detection_rate']:.1%}{'':>9} {result['avg_detections_per_frame']:<15.2f} {result['avg_good_keypoints']:<12.1f} {result['avg_confidence']:<10.3f} {result['fps']:<8.1f}")
        else:
            print(f"{name:<30} {'ERROR':<15}")
    
    print(f"\n🏆 RECOMMENDATIONS:")
    
    # Find best models
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if valid_results:
        best_detection_rate = max(valid_results.items(), key=lambda x: x[1]['detection_rate'])
        best_confidence = max(valid_results.items(), key=lambda x: x[1]['avg_confidence'])
        best_fps = max(valid_results.items(), key=lambda x: x[1]['fps'])
        
        print(f"✅ Best detection rate: {best_detection_rate[0]} ({best_detection_rate[1]['detection_rate']:.1%})")
        print(f"✅ Best confidence: {best_confidence[0]} ({best_confidence[1]['avg_confidence']:.3f})")
        print(f"✅ Best speed: {best_fps[0]} ({best_fps[1]['fps']:.1f} fps)")
        
        # Overall recommendation
        improved_11 = results.get('Improved Models (11-keypoint)')
        improved_4 = results.get('Improved Detection + 4-keypoint')
        
        if improved_11 and improved_4:
            if improved_11['avg_confidence'] > improved_4['avg_confidence']:
                print(f"\n🎯 OVERALL RECOMMENDATION: Improved Models (11-keypoint)")
                print(f"   ✅ Higher confidence and more detailed pose information")
            else:
                print(f"\n🎯 OVERALL RECOMMENDATION: Improved Detection + 4-keypoint")
                print(f"   ✅ Good balance of reliability and performance")


def main():
    parser = argparse.ArgumentParser(description='Test Video Model Combinations')
    parser.add_argument('video', help='Input video path')
    parser.add_argument('--frames', type=int, default=100, help='Number of frames to test')
    
    args = parser.parse_args()
    
    test_video_models(args.video, args.frames)


if __name__ == '__main__':
    main()
