#!/usr/bin/env python3
"""
Integrated silkworm detection + pose estimation for video inference
Uses your Faster R-CNN model for detection, then MMPose for pose estimation
"""

import os
import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import argparse
from tqdm import tqdm

from mmpose.apis import init_model, inference_topdown
from mmpose.visualization import PoseLocalVisualizer


class SilkwormDetector:
    """Wrapper for your Faster R-CNN silkworm detection model"""
    
    def __init__(self, model_path, device='cuda:0', num_classes=5):
        self.device = torch.device(device)
        self.model = self._load_model(model_path, num_classes)
        self.transform = transforms.Compose([transforms.ToTensor()])

        # Class mapping for new model (extend strategy)
        self.class_names = {0: "background", 1: "kaiko_live", 2: "kaiko_pao1", 3: "kaiko_rhi", 4: "kaiko"}
        
    def _load_model(self, model_path, num_classes):
        """Load the Faster R-CNN model with your trained weights"""
        # Create model architecture (same as your get_model function)
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Load your trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval()
        
        return model
    
    def detect(self, image, score_threshold=0.7):
        """
        Detect silkworms in image
        Args:
            image: numpy array (BGR format from cv2) or PIL Image
            score_threshold: minimum confidence score
        Returns:
            bboxes: numpy array of shape (N, 4) with [x1, y1, x2, y2] format
            scores: numpy array of confidence scores
            labels: numpy array of class labels
        """
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
        else:
            pil_image = image
        
        # Preprocess
        img_tensor = self.transform(pil_image).to(self.device)
        
        # Run detection
        with torch.no_grad():
            outputs = self.model([img_tensor])
        
        output = outputs[0]
        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        
        # Filter by score threshold and exclude background (label 0)
        valid_mask = (scores >= score_threshold) & (labels != 0)
        
        return boxes[valid_mask], scores[valid_mask], labels[valid_mask]


def process_video_with_detection(detection_model, pose_model, input_path, output_path, 
                                detection_threshold=0.7, pose_threshold=0.2):
    """Process video with detection + pose estimation"""
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {width}x{height}, {fps}fps, {total_frames} frames")
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize pose visualizer
    pose_visualizer = PoseLocalVisualizer()
    pose_visualizer.set_dataset_meta(pose_model.dataset_meta)
    
    frame_count = 0
    detection_stats = {"total_detections": 0, "frames_with_detections": 0}
    
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # 1. Detect silkworms
                bboxes, scores, labels = detection_model.detect(frame, detection_threshold)
                
                if len(bboxes) > 0:
                    detection_stats["frames_with_detections"] += 1
                    detection_stats["total_detections"] += len(bboxes)
                    
                    # 2. Run pose estimation on ALL detected silkworms
                    # Convert bboxes to the format expected by MMPose
                    mmpose_bboxes = []
                    for bbox in bboxes:
                        # bbox is [x1, y1, x2, y2], convert to [x1, y1, x2, y2] (same format)
                        mmpose_bboxes.append(bbox.astype(np.float32))

                    # Draw detection boxes first
                    vis_frame = frame.copy()
                    for i, (bbox, score, label) in enumerate(zip(bboxes, scores, labels)):
                        x1, y1, x2, y2 = bbox.astype(int)
                        class_name = detection_model.class_names.get(label, f"class_{label}")

                        # Draw bounding box with color coding
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
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

                        # Draw label with silkworm ID
                        label_text = f"ID{i+1} {class_name}: {score:.2f}"
                        cv2.putText(vis_frame, label_text, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # 3. Run pose estimation on ALL silkworms
                    if mmpose_bboxes:
                        pose_preds = inference_topdown(pose_model, frame, mmpose_bboxes)

                        # 4. Draw pose for each silkworm individually
                        if len(pose_preds) > 0:
                            for i, pred in enumerate(pose_preds):
                                if i < len(bboxes):  # Make sure we have corresponding detection
                                    # Get keypoints and scores
                                    keypoints = pred.pred_instances.keypoints[0]
                                    keypoint_scores = pred.pred_instances.keypoint_scores[0]

                                    # Use same color as detection box
                                    label = labels[i]
                                    if label == 1:
                                        kpt_color = (0, 255, 0)      # Green
                                    elif label == 2:
                                        kpt_color = (0, 0, 255)      # Red
                                    elif label == 3:
                                        kpt_color = (0, 165, 255)    # Orange
                                    elif label == 4:
                                        kpt_color = (255, 0, 0)      # Blue
                                    else:
                                        kpt_color = (128, 128, 128)  # Gray

                                    # Draw keypoints manually for better control
                                    for j, (kpt, kpt_score) in enumerate(zip(keypoints, keypoint_scores)):
                                        if kpt_score > pose_threshold:
                                            x, y = int(kpt[0]), int(kpt[1])
                                            cv2.circle(vis_frame, (x, y), 4, kpt_color, -1)
                                            cv2.circle(vis_frame, (x, y), 6, (255, 255, 255), 2)  # White border
                                            # Draw keypoint index
                                            cv2.putText(vis_frame, f"{j}", (x+8, y-8),
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, kpt_color, 1)

                                    # Draw skeleton connections
                                    skeleton = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10]]
                                    for connection in skeleton:
                                        if len(keypoints) > max(connection):
                                            pt1_idx, pt2_idx = connection
                                            if (keypoint_scores[pt1_idx] > pose_threshold and
                                                keypoint_scores[pt2_idx] > pose_threshold):
                                                pt1 = (int(keypoints[pt1_idx][0]), int(keypoints[pt1_idx][1]))
                                                pt2 = (int(keypoints[pt2_idx][0]), int(keypoints[pt2_idx][1]))
                                                cv2.line(vis_frame, pt1, pt2, kpt_color, 3)
                    else:
                        vis_frame = frame
                else:
                    # No detections, use original frame
                    vis_frame = frame
                    
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                vis_frame = frame
            
            # Write frame
            out.write(vis_frame)
            frame_count += 1
            pbar.update(1)
            
            # Print progress every 100 frames
            if frame_count % 100 == 0:
                avg_detections = detection_stats["total_detections"] / max(1, detection_stats["frames_with_detections"])
                print(f"Frame {frame_count}: {detection_stats['frames_with_detections']} frames with detections, "
                      f"avg {avg_detections:.1f} silkworms per frame")
    
    cap.release()
    out.release()
    
    return detection_stats


def main():
    parser = argparse.ArgumentParser(description='Silkworm Detection + Pose Estimation')
    parser.add_argument('input_video', help='Input video file')
    parser.add_argument('--output', '-o', help='Output video path')
    
    # Detection model args
    parser.add_argument('--detection-model',
                       default='work_dirs/detection_training_extend/final_model.pth',
                       help='Path to detection model weights')
    parser.add_argument('--detection-threshold', type=float, default=0.7,
                       help='Detection confidence threshold')
    
    # Pose model args  
    parser.add_argument('--pose-config',
                       default='configs/silkworm/td-hm_hrnet-w32_8xb64-210e_silkworm-11keypoints-expanded.py',
                       help='Pose estimation config file')
    parser.add_argument('--pose-checkpoint',
                       default='work_dirs/hrnet_silkworm_11keypoints_expanded/best_coco_AP_epoch_200.pth',
                       help='Pose estimation checkpoint')
    parser.add_argument('--pose-threshold', type=float, default=0.2,
                       help='Pose keypoint confidence threshold')
    
    parser.add_argument('--device', default='cuda:0', help='Device to use')
    
    args = parser.parse_args()
    
    # Set output path
    if args.output is None:
        base_name = os.path.splitext(args.input_video)[0]
        args.output = f"{base_name}_detection_pose.mp4"
    
    print("🔄 Loading detection model...")
    detection_model = SilkwormDetector(args.detection_model, args.device)
    print("✅ Detection model loaded!")
    
    print("🔄 Loading pose estimation model...")
    pose_model = init_model(args.pose_config, args.pose_checkpoint, device=args.device)
    print("✅ Pose model loaded!")
    
    print(f"🎬 Processing video: {args.input_video}")
    print(f"📁 Output will be saved to: {args.output}")
    
    try:
        stats = process_video_with_detection(
            detection_model, pose_model, 
            args.input_video, args.output,
            args.detection_threshold, args.pose_threshold
        )
        
        print(f"\n✅ Video processing completed!")
        print(f"📊 Statistics:")
        print(f"   - Frames with detections: {stats['frames_with_detections']}")
        print(f"   - Total detections: {stats['total_detections']}")
        print(f"   - Average detections per frame: {stats['total_detections']/max(1, stats['frames_with_detections']):.1f}")
        print(f"💾 Output saved to: {os.path.abspath(args.output)}")
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")


if __name__ == '__main__':
    main()
