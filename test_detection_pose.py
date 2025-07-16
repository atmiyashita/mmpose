#!/usr/bin/env python3
"""
Simple test script for detection + pose estimation on a single image
"""

import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import argparse

from mmpose.apis import init_model, inference_topdown
from mmpose.visualization import PoseLocalVisualizer


def load_detection_model(model_path, device='cuda:0', num_classes=5):
    """Load your Faster R-CNN detection model"""
    device = torch.device(device)

    # Create model (same as your code)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model, device


def detect_silkworms(model, device, image, score_threshold=0.7):
    """Run detection on image"""
    # Convert to PIL and preprocess
    if isinstance(image, np.ndarray):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
    else:
        pil_image = image
    
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(pil_image).to(device)
    
    # Run detection
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
    parser = argparse.ArgumentParser(description='Test Detection + Pose on Single Image')
    parser.add_argument('input_image', help='Input image file')
    parser.add_argument('--output', '-o', help='Output image path')
    parser.add_argument('--detection-model',
                       default='work_dirs/detection_training_extend/final_model.pth')
    parser.add_argument('--pose-config',
                       default='configs/silkworm/td-hm_hrnet-w32_8xb64-210e_silkworm-4keypoints.py')
    parser.add_argument('--pose-checkpoint',
                       default='work_dirs/hrnet_silkworm_4keypoints/best_coco_AP_epoch_130.pth')
    parser.add_argument('--device', default='cuda:0')
    
    args = parser.parse_args()
    
    if args.output is None:
        base_name = args.input_image.rsplit('.', 1)[0]
        args.output = f"{base_name}_detection_pose.jpg"
    
    print("🔄 Loading models...")
    
    # Load detection model
    detection_model, device = load_detection_model(args.detection_model, args.device)
    print("✅ Detection model loaded!")
    
    # Load pose model
    pose_model = init_model(args.pose_config, args.pose_checkpoint, device=args.device)
    print("✅ Pose model loaded!")
    
    # Load image
    image = cv2.imread(args.input_image)
    if image is None:
        print(f"❌ Could not load image: {args.input_image}")
        return
    
    print(f"📷 Processing image: {args.input_image}")
    
    # 1. Run detection
    bboxes, scores, labels = detect_silkworms(detection_model, device, image)
    print(f"🔍 Detected {len(bboxes)} silkworms")
    
    class_names = {1: "kaiko_live", 2: "kaiko_pao1", 3: "kaiko_rhi", 4: "kaiko"}
    
    if len(bboxes) > 0:
        # Print detection results
        for i, (bbox, score, label) in enumerate(zip(bboxes, scores, labels)):
            class_name = class_names.get(label, f"class_{label}")
            print(f"  Silkworm {i+1}: {class_name} (conf: {score:.3f})")
        
        # 2. Run pose estimation
        mmpose_bboxes = [bbox.astype(np.float32) for bbox in bboxes]
        pose_preds = inference_topdown(pose_model, image, mmpose_bboxes)
        print(f"🎯 Generated {len(pose_preds)} pose predictions")
        
        # 3. Visualize results
        vis_image = image.copy()
        
        # Draw detection boxes
        for bbox, score, label in zip(bboxes, scores, labels):
            x1, y1, x2, y2 = bbox.astype(int)
            class_name = class_names.get(label, f"class_{label}")
            
            # Color coding: live=green, pao1=red, rhi=orange, kaiko=blue
            if label == 1:
                color = (0, 255, 0)      # Green for kaiko_live
            elif label == 2:
                color = (0, 0, 255)      # Red for kaiko_pao1
            elif label == 3:
                color = (0, 165, 255)    # Orange for kaiko_rhi
            elif label == 4:
                color = (255, 0, 0)      # Blue for kaiko (your new data)
            else:
                color = (128, 128, 128)  # Gray for unknown
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label_text = f"{class_name}: {score:.2f}"
            cv2.putText(vis_image, label_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add pose visualization
        if len(pose_preds) > 0:
            visualizer = PoseLocalVisualizer()
            visualizer.set_dataset_meta(pose_model.dataset_meta)
            visualizer.set_image(vis_image)
            
            for pred in pose_preds:
                visualizer.add_datasample(
                    'silkworm',
                    vis_image,
                    data_sample=pred,
                    draw_gt=False,
                    draw_heatmap=False,
                    show_kpt_idx=True,
                    skeleton_style='mmpose',
                    kpt_thr=0.2
                )
            
            vis_image = visualizer.get_image()
        
        # Save result
        cv2.imwrite(args.output, vis_image)
        print(f"✅ Result saved to: {args.output}")
        
        # Print pose info
        if len(pose_preds) > 0:
            keypoint_names = ['H1', 'T1', 'T2', 'T3', 'A2', 'A3', 'A4', 'A5', 'A6', 'A8', 'A9']
            for i, pred in enumerate(pose_preds):
                print(f"\n🐛 Silkworm {i+1} pose keypoints:")
                pred_instances = pred.pred_instances
                keypoints = pred_instances.keypoints[0]
                scores = pred_instances.keypoint_scores[0]
                
                for j, (kpt, score) in enumerate(zip(keypoints, scores)):
                    if score > 0.2:
                        print(f"  {keypoint_names[j]:3s}: ({kpt[0]:6.1f}, {kpt[1]:6.1f}) conf={score:.3f}")
    else:
        print("❌ No silkworms detected!")
        cv2.imwrite(args.output, image)


if __name__ == '__main__':
    main()
