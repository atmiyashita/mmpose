#!/usr/bin/env python3
"""
Multi-silkworm video inference with detection + pose estimation
"""

import os
import cv2
import numpy as np
import argparse
from mmpose.apis import init_model, inference_topdown
from mmpose.visualization import PoseLocalVisualizer


def detect_silkworms_simple(frame, min_area=1000):
    """
    Simple silkworm detection using contour detection
    Returns bounding boxes for detected silkworms
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to separate silkworms from background
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for contour in contours:
        # Filter by area
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            # Add some padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)
            
            bboxes.append([x, y, x + w, y + h])
    
    return np.array(bboxes, dtype=np.float32) if bboxes else np.array([]).reshape(0, 4)


def detect_silkworms_advanced(frame):
    """
    More advanced silkworm detection using morphological operations
    """
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for silkworm colors (adjust based on your silkworms)
    # This is a general range - you may need to tune these values
    lower_bound = np.array([0, 0, 50])    # Lower HSV bound
    upper_bound = np.array([180, 50, 200]) # Upper HSV bound
    
    # Create mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 2000:  # Minimum area threshold
            x, y, w, h = cv2.boundingRect(contour)
            # Add padding
            padding = 30
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)
            
            bboxes.append([x, y, x + w, y + h])
    
    return np.array(bboxes, dtype=np.float32) if bboxes else np.array([]).reshape(0, 4)


def main():
    parser = argparse.ArgumentParser(description='Multi-Silkworm Video Inference')
    parser.add_argument('input_video', help='Input video file')
    parser.add_argument('--output', '-o', help='Output video path')
    parser.add_argument('--config', 
                       default='configs/silkworm/td-hm_hrnet-w32_8xb64-210e_silkworm-256x192.py')
    parser.add_argument('--checkpoint', 
                       default='work_dirs/hrnet_silkworm11/epoch_210.pth')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--detection-method', choices=['simple', 'advanced', 'none'], 
                       default='simple', help='Silkworm detection method')
    parser.add_argument('--min-area', type=int, default=1000,
                       help='Minimum area for silkworm detection')
    
    args = parser.parse_args()
    
    if args.output is None:
        base_name = os.path.splitext(args.input_video)[0]
        args.output = f"{base_name}_multi_pose.mp4"
    
    print("🔄 Loading pose estimation model...")
    model = init_model(args.config, args.checkpoint, device=args.device)
    print("✅ Model loaded!")
    
    # Open video
    cap = cv2.VideoCapture(args.input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {width}x{height}, {fps}fps, {total_frames} frames")
    
    # Setup output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Initialize visualizer
    visualizer = PoseLocalVisualizer()
    visualizer.set_dataset_meta(model.dataset_meta)
    
    frame_num = 0
    print("🎬 Processing video...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect silkworms
        if args.detection_method == 'simple':
            bboxes = detect_silkworms_simple(frame, args.min_area)
        elif args.detection_method == 'advanced':
            bboxes = detect_silkworms_advanced(frame)
        else:  # none - use entire frame
            bboxes = np.array([[0, 0, width, height]], dtype=np.float32)
        
        if len(bboxes) == 0:
            # No silkworms detected, use original frame
            result_frame = frame
        else:
            # Run pose estimation on detected silkworms
            preds = inference_topdown(model, frame, bboxes)
            
            if len(preds) > 0:
                # Visualize all detected silkworms
                visualizer.set_image(frame)
                for i, pred in enumerate(preds):
                    visualizer.add_datasample(
                        f'silkworm_{i}',
                        frame,
                        data_sample=pred,
                        draw_gt=False,
                        draw_heatmap=False,
                        show_kpt_idx=False,
                        skeleton_style='mmpose',
                        kpt_thr=0.2
                    )
                result_frame = visualizer.get_image()
            else:
                result_frame = frame
        
        out.write(result_frame)
        frame_num += 1
        
        if frame_num % 30 == 0:
            progress = (frame_num / total_frames) * 100
            detected = len(bboxes) if len(bboxes) > 0 else 0
            print(f"⏳ Frame {frame_num}/{total_frames} ({progress:.1f}%) - {detected} silkworms detected")
    
    cap.release()
    out.release()
    
    print(f"✅ Done! Output saved to: {args.output}")
    print(f"📊 Processed {frame_num} frames")


if __name__ == '__main__':
    main()
