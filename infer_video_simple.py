#!/usr/bin/env python3
"""
Simple video inference script for silkworm pose estimation
"""

import os
import cv2
import numpy as np
import argparse
from mmpose.apis import init_model, inference_topdown
from mmpose.visualization import PoseLocalVisualizer


def main():
    parser = argparse.ArgumentParser(description='Simple Silkworm Video Inference')
    parser.add_argument('input_video', help='Input video file path')
    parser.add_argument('--output', '-o', help='Output video path')
    parser.add_argument('--config', 
                       default='configs/silkworm/td-hm_hrnet-w32_8xb64-210e_silkworm-256x192.py')
    parser.add_argument('--checkpoint', 
                       default='work_dirs/hrnet_silkworm11/epoch_210.pth')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--conf-thr', type=float, default=0.2,
                       help='Confidence threshold for keypoints')
    
    args = parser.parse_args()
    
    # Set output path
    if args.output is None:
        base_name = os.path.splitext(args.input_video)[0]
        args.output = f"{base_name}_pose.mp4"
    
    # Check input exists
    if not os.path.exists(args.input_video):
        print(f"❌ Input video not found: {args.input_video}")
        return
    
    print("🔄 Loading model...")
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
    
    # Process frames
    frame_num = 0
    person_bbox = [np.array([0, 0, width, height], dtype=np.float32)]
    
    print("🎬 Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run inference
        preds = inference_topdown(model, frame, person_bbox)
        
        if len(preds) > 0:
            # Visualize
            visualizer.set_image(frame)
            visualizer.add_datasample(
                'silkworm',
                frame,
                data_sample=preds[0],
                draw_gt=False,
                draw_heatmap=False,
                show_kpt_idx=False,
                skeleton_style='mmpose',
                kpt_thr=args.conf_thr
            )
            result_frame = visualizer.get_image()
        else:
            result_frame = frame
        
        out.write(result_frame)
        frame_num += 1
        
        # Progress indicator
        if frame_num % 30 == 0:
            progress = (frame_num / total_frames) * 100
            print(f"⏳ Progress: {frame_num}/{total_frames} ({progress:.1f}%)")
    
    cap.release()
    out.release()
    
    print(f"✅ Done! Output saved to: {args.output}")
    print(f"📊 Processed {frame_num} frames")


if __name__ == '__main__':
    main()
